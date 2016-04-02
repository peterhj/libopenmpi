use libc::{c_void, c_char, c_int, c_longlong};

#[repr(C)]
pub struct MPI_Offset(c_longlong);

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Count(c_longlong);

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct MPI_Status {
  pub source:       c_int,
  pub tag:          c_int,
  pub error:        c_int,
  pub count:        MPI_Count,
  pub cancelled:    c_int,
  pub abi_slush_fund:   [c_int; 2],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Datatype(c_int);

pub const MPI_CHAR:                 MPI_Datatype = MPI_Datatype(0x4c000101);
pub const MPI_SIGNED_CHAR:          MPI_Datatype = MPI_Datatype(0x4c000118);
pub const MPI_UNSIGNED_CHAR:        MPI_Datatype = MPI_Datatype(0x4c000102);
pub const MPI_BYTE:                 MPI_Datatype = MPI_Datatype(0x4c00010d);
pub const MPI_LONG:                 MPI_Datatype = MPI_Datatype(0x4c000807);
pub const MPI_UNSIGNED_LONG:        MPI_Datatype = MPI_Datatype(0x4c000808);
pub const MPI_LONG_LONG_INT:        MPI_Datatype = MPI_Datatype(0x4c000809);
pub const MPI_UNSIGNED_LONG_LONG:   MPI_Datatype = MPI_Datatype(0x4c000819);
pub const MPI_FLOAT:                MPI_Datatype = MPI_Datatype(0x4c00040a);
pub const MPI_DOUBLE:               MPI_Datatype = MPI_Datatype(0x4c00080b);

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MPI_Comm(c_int);

pub const MPI_COMM_WORLD:   MPI_Comm = MPI_Comm(0x44000000);
pub const MPI_COMM_SELF:    MPI_Comm = MPI_Comm(0x44000001);

#[repr(C)]
pub struct MPI_Group(c_int);

pub const MPI_GROUP_EMPTY:  MPI_Group = MPI_Group(0x48000000);

#[repr(C)]
pub struct MPI_Win(c_int);

pub const MPI_WIN_NULL:     MPI_Win = MPI_Win(0x20000000);

#[repr(C)]
pub struct MPI_Op(c_int);

pub const MPI_MAX:          MPI_Op = MPI_Op(0x58000001);
pub const MPI_MIN:          MPI_Op = MPI_Op(0x58000002);
pub const MPI_SUM:          MPI_Op = MPI_Op(0x58000003);
pub const MPI_REPLACE:      MPI_Op = MPI_Op(0x5800000d);
pub const MPI_NO_OP:        MPI_Op = MPI_Op(0x5800000e);

#[link(name = "dl")]
extern "C" {
}

#[link(name = "hwloc")]
extern "C" {
}

#[link(name = "mpi")]
extern "C" {
  pub fn MPI_Init(argc: *mut c_int, argv: *mut *mut *mut c_char) -> c_int;
  pub fn MPI_Initialized() -> c_int;
  pub fn MPI_Finalize() -> c_int;
  pub fn MPI_Abort(comm: MPI_Comm, errorcode: c_int) -> c_int;

  pub fn MPI_Comm_size(comm: MPI_Comm, size: *mut c_int) -> c_int;
  pub fn MPI_Comm_rank(comm: MPI_Comm, rank: *mut c_int) -> c_int;

  pub fn MPI_Send(buf: *const c_void, count: c_int, datatype: MPI_Datatype, dest: c_int, tag: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Recv(buf: *mut c_void, count: c_int, datatype: MPI_Datatype, source: c_int, tag: c_int, comm: MPI_Comm, status: *mut MPI_Status) -> c_int;

  pub fn MPI_Barrier(comm: MPI_Comm) -> c_int;
  pub fn MPI_Bcast(buf: *const c_void, count: c_int, datatype: MPI_Datatype, root: c_int, comm: MPI_Comm) -> c_int;
  pub fn MPI_Allreduce(sendbuf: *const c_void, recvbuf: *mut c_void, count: c_int, datatype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> c_int;
}
