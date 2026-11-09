import RoomPlan
import SceneKit
import simd

/// RoomPlan's `rooms/room-N.json` reduced to what LiDAR coverage needs, in the session ARKit frame.
struct RoomLayout {
  struct Box {
    let transform: simd_float4x4
    let inverse: simd_float4x4
    let size: SIMD3<Float>
    func local(_ p: SIMD3<Float>) -> SIMD3<Float> { simd_make_float3(inverse * SIMD4(p, 1)) }
  }

  init() {}

  var walls: [Box] = []
  /// doors, windows and openings: real holes in the mesh nobody can fill
  var cutouts: [Box] = []
  var objects: [Box] = []
  var floorY: Float?
  /// floor outline on the plan (world x, z)
  var outline: [SIMD2<Float>] = []

  static func load(_ url: URL) -> RoomLayout? {
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    func boxes(_ keys: [String]) -> [Box] {
      keys.flatMap { json[$0] as? [[String: Any]] ?? [] }.compactMap { d in
        guard let t = d["transform"] as? [Float], t.count == 16,
              let dim = d["dimensions"] as? [Float], dim.count >= 2 else { return nil }
        let m = simd_float4x4(columnMajor: t)
        return Box(transform: m, inverse: m.inverse, size: SIMD3(dim[0], dim[1], dim.count > 2 ? dim[2] : 0))
      }
    }
    var layout = RoomLayout()
    layout.walls = boxes(["walls"])
    layout.cutouts = boxes(["doors", "windows", "openings"])
    layout.objects = boxes(["objects"])
    if let floor = (json["floors"] as? [[String: Any]])?.first,
       let t = floor["transform"] as? [Float], t.count == 16,
       let corners = floor["polygonCorners"] as? [[Float]], corners.count >= 3 {
      let m = simd_float4x4(columnMajor: t)
      layout.floorY = m.columns.3.y
      layout.outline = corners.compactMap { c in
        guard c.count >= 2 else { return nil }
        let w = m * SIMD4(c[0], c[1], c.count > 2 ? c[2] : 0, 1)
        return SIMD2(w.x, w.z)
      }
    }
    return layout.walls.isEmpty && layout.outline.isEmpty ? nil : layout
  }
}

extension RoomLayout {
  /// RoomPlan's live room; the floor outline falls back to the walls' extent once there are 3 walls
  init?(room: CapturedRoom) {
    func box(_ s: CapturedRoom.Surface) -> Box { Box(transform: s.transform, inverse: s.transform.inverse, size: s.dimensions) }
    walls = room.walls.map(box)
    cutouts = (room.doors + room.windows + room.openings).map(box)
    objects = room.objects.map { Box(transform: $0.transform, inverse: $0.transform.inverse, size: $0.dimensions) }
    if let floor = room.floors.first, !floor.polygonCorners.isEmpty {
      floorY = floor.transform.columns.3.y
      outline = floor.polygonCorners.map { c in
        let w = floor.transform * SIMD4(c, 1)
        return SIMD2(w.x, w.z)
      }
    } else if walls.count >= 3 {
      outline = CoverageEngine.bounds(for: room).0
    }
    if walls.isEmpty && outline.isEmpty { return nil }
  }
}

/// World-frame triangles of a mesh, whatever stores them
protocol TriangleSource {
  /// Calls `visit` for triangles that may touch the box; `visit` returns false to stop.
  func forEachTriangle(within lo: SIMD3<Float>, _ hi: SIMD3<Float>,
                       _ visit: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) -> Bool)
}

extension PLYMesh: TriangleSource {
  func forEachTriangle(within lo: SIMD3<Float>, _ hi: SIMD3<Float>,
                       _ visit: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) -> Bool) {
    var f = 0
    while f + 2 < indices.count {
      if !visit(position(Int(indices[f])), position(Int(indices[f + 1])), position(Int(indices[f + 2]))) { return }
      f += 3
    }
  }
}

/// A snapshot of the live scan's mesh anchors, in the ARKit world frame
struct LiveMesh: TriangleSource {
  struct Part {
    var world: [SIMD3<Float>]
    var faces: [UInt32]
    var lo: SIMD3<Float>
    var hi: SIMD3<Float>
  }
  var parts: [Part] = []

  func forEachTriangle(within lo: SIMD3<Float>, _ hi: SIMD3<Float>,
                       _ visit: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) -> Bool) {
    for part in parts where !(any(part.hi .< lo) || any(part.lo .> hi)) {
      var f = 0
      while f + 2 < part.faces.count {
        if !visit(part.world[Int(part.faces[f])], part.world[Int(part.faces[f + 1])], part.world[Int(part.faces[f + 2])]) { return }
        f += 3
      }
    }
  }
}

/// Which parts of a room's walls, floor and ceiling the LiDAR mesh actually reaches. The mesh only
/// exists where LiDAR saw something, so holes are found against RoomPlan's surfaces.
enum LidarCoverage {
  struct Result {
    /// orange quads over the uncovered cells (nil when nothing is missing)
    let gaps: SCNGeometry?
    /// areas in m²
    let covered: Float
    let total: Float
  }

  private struct Sample {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
  }

  static let cell: Float = 0.10
  private static let doorMargin: Float = 0.05
  private static let objectMargin: Float = 0.10

  /// nil when cancelled or when the layout has no surface to test
  static func compute(mesh: some TriangleSource, layout: RoomLayout, toHouse: simd_float4x4,
                      isCancelled: () -> Bool) -> Result? {
    let samples = samples(of: layout).map { s in
      Sample(position: simd_make_float3(toHouse * SIMD4(s.position, 1)),
             normal: simd_normalize(simd_make_float3(toHouse * SIMD4(s.normal, 0))))
    }
    guard let first = samples.first else { return nil }
    var lo = first.position, hi = first.position
    for s in samples { lo = simd_min(lo, s.position); hi = simd_max(hi, s.position) }
    let reach = SurroundingsFilter.voxel * 2
    guard let occupied = occupancy(mesh, lo - reach, hi + reach, isCancelled) else { return nil }

    var positions: [SCNVector3] = [], normals: [SCNVector3] = [], indices: [UInt32] = []
    var uncovered = 0
    for s in samples {
      if isNear(s.position, occupied) { continue }
      uncovered += 1
      let up: SIMD3<Float> = abs(s.normal.y) > 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 1, 0)
      let u = simd_normalize(simd_cross(s.normal, up)) * (cell / 2)
      let v = simd_normalize(simd_cross(s.normal, u)) * (cell / 2)
      let base = UInt32(positions.count)
      for corner in [s.position - u - v, s.position + u - v, s.position + u + v, s.position - u + v] {
        positions.append(SCNVector3(corner.x, corner.y, corner.z))
        normals.append(SCNVector3(s.normal.x, s.normal.y, s.normal.z))
      }
      indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    }
    let area = cell * cell
    let total = Float(samples.count) * area
    var geometry: SCNGeometry?
    if !positions.isEmpty {
      geometry = SCNGeometry(sources: [SCNGeometrySource(vertices: positions), SCNGeometrySource(normals: normals)],
                             elements: [SCNGeometryElement(indices: indices, primitiveType: .triangles)])
    }
    return Result(gaps: geometry, covered: total - Float(uncovered) * area, total: total)
  }

  // MARK: reference surfaces

  /// One sample per 10 cm cell on walls, floor and ceiling, minus what a door or furniture hides
  private static func samples(of layout: RoomLayout) -> [Sample] {
    var out: [Sample] = []
    for wall in layout.walls {
      let t = wall.transform
      let x = simd_make_float3(t.columns.0), y = simd_make_float3(t.columns.1)
      let c = simd_make_float3(t.columns.3), n = simd_normalize(simd_make_float3(t.columns.2))
      let nu = max(1, Int((wall.size.x / cell).rounded())), nv = max(1, Int((wall.size.y / cell).rounded()))
      for i in 0..<nu {
        for j in 0..<nv {
          let p = c + x * ((Float(i) + 0.5) / Float(nu) - 0.5) * wall.size.x
                    + y * ((Float(j) + 0.5) / Float(nv) - 0.5) * wall.size.y
          if hidden(p, layout, byCutouts: true) { continue }
          out.append(Sample(position: p, normal: n))
        }
      }
    }
    guard layout.outline.count >= 3 else { return out }
    let wallTops = layout.walls.map { $0.transform.columns.3.y + $0.size.y / 2 }
    let floorY = layout.floorY ?? layout.walls.map { $0.transform.columns.3.y - $0.size.y / 2 }.min() ?? 0
    let lo = layout.outline.reduce(layout.outline[0]) { simd_min($0, $1) }
    let hi = layout.outline.reduce(layout.outline[0]) { simd_max($0, $1) }
    var x = lo.x + cell / 2
    while x < hi.x {
      var z = lo.y + cell / 2
      while z < hi.y {
        if CoverageEngine.inside(layout.outline, SIMD2(x, z)) {
          let floor = SIMD3<Float>(x, floorY, z)
          if !hidden(floor, layout, byCutouts: false) { out.append(Sample(position: floor, normal: SIMD3(0, 1, 0))) }
          if let top = wallTops.max() {
            let ceiling = SIMD3<Float>(x, top, z)
            if !hidden(ceiling, layout, byCutouts: false) { out.append(Sample(position: ceiling, normal: SIMD3(0, -1, 0))) }
          }
        }
        z += cell
      }
      x += cell
    }
    return out
  }

  private static func hidden(_ p: SIMD3<Float>, _ layout: RoomLayout, byCutouts: Bool) -> Bool {
    if byCutouts {
      for c in layout.cutouts {
        let q = c.local(p)
        if abs(q.x) < c.size.x / 2 + doorMargin, abs(q.y) < c.size.y / 2 + doorMargin, abs(q.z) < 0.3 { return true }
      }
    }
    for o in layout.objects {
      let q = o.local(p), h = o.size / 2 + objectMargin
      if abs(q.x) < h.x, abs(q.y) < h.y, abs(q.z) < h.z { return true }
    }
    return false
  }

  // MARK: mesh occupancy

  /// Voxels touched by the mesh triangles inside the box. Triangles are filled, not just their
  /// centroids: flat floors use triangles far larger than a voxel.
  private static func occupancy(_ mesh: some TriangleSource, _ lo: SIMD3<Float>, _ hi: SIMD3<Float>,
                                _ isCancelled: () -> Bool) -> Set<Int64>? {
    var set = Set<Int64>()
    let step = SurroundingsFilter.voxel
    var seen = 0
    var stopped = false
    mesh.forEachTriangle(within: lo, hi) { a, b, c in
      seen += 1
      if seen & 0x3FFF == 0, isCancelled() { stopped = true; return false }
      let tLo = simd_min(a, simd_min(b, c)), tHi = simd_max(a, simd_max(b, c))
      if any(tHi .< lo) || any(tLo .> hi) { return true }
      let longest = max(simd_length(b - a), simd_length(c - b), simd_length(a - c))
      let n = min(32, max(1, Int((longest / step).rounded(.up))))
      for i in 0...n {
        for j in 0...(n - i) {
          set.insert(SurroundingsFilter.key(a + (b - a) * (Float(i) / Float(n)) + (c - a) * (Float(j) / Float(n))))
        }
      }
      return true
    }
    if stopped { return nil }
    return set
  }

  /// true when mesh lies in the voxel of `p` or one of its 26 neighbours (about 8 to 16 cm)
  private static func isNear(_ p: SIMD3<Float>, _ set: Set<Int64>) -> Bool {
    let step = SurroundingsFilter.voxel
    for dx in -1...1 {
      for dy in -1...1 {
        for dz in -1...1 where set.contains(SurroundingsFilter.key(p + SIMD3(Float(dx), Float(dy), Float(dz)) * step)) {
          return true
        }
      }
    }
    return false
  }
}
