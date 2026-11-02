import type { RoomModule } from '../packages/room-runtime/types';
export const ROOMS: RoomModule[] = [{id:'office-v4', label:'Wes’s office', build:async () => (await import('./office-v4')).officeV4.build()},
  {id:'atrium', label:'Atrium', build:async () => (await import('./atrium')).atrium.build()},
  {id:'trone', label:'Trone', build:async () => (await import('./trone')).trone.build()},
  {id:'couloir', label:'Couloir', build:async () => (await import('./couloir')).couloir.build()}];
export const roomById = (id:string|null) => ROOMS.find(r=>r.id===id) ?? ROOMS[0];
export type {BuiltRoom, RoomModule} from '../packages/room-runtime/types';
