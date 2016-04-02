extern crate libc;

use ffi::*;

use libc::{c_void, c_int};
use std::env;
use std::ffi::{CString};

pub mod ffi;

pub trait MpiData {
  fn datatype() -> MPI_Datatype;
}

impl MpiData for u8 {
  fn datatype() -> MPI_Datatype {
    MPI_BYTE
  }
}

impl MpiData for u64 {
  fn datatype() -> MPI_Datatype {
    MPI_UNSIGNED_LONG_LONG
  }
}

impl MpiData for f32 {
  fn datatype() -> MPI_Datatype {
    MPI_FLOAT
  }
}

impl MpiData for f64 {
  fn datatype() -> MPI_Datatype {
    MPI_DOUBLE
  }
}

pub trait MpiOp {
  fn op() -> MPI_Op;
}

pub struct MpiSumOp;

impl MpiOp for MpiSumOp {
  fn op() -> MPI_Op {
    MPI_SUM
  }
}

pub struct Mpi;

impl Drop for Mpi {
  fn drop(&mut self) {
    unsafe { MPI_Finalize() };
  }
}

impl Mpi {
  pub fn new() -> Mpi {
    let args: Vec<_> = env::args().collect();
    // FIXME(20160130): this leaks the C string.
    let mut c_args: Vec<_> = args.into_iter().map(|s| match CString::new(s) {
      Ok(s) => s.into_raw(),
      Err(e) => panic!("mpi: failed to initialize: bad argv: {:?}", e),
    }).collect();
    let mut argc = c_args.len() as c_int;
    let mut argv = (&mut c_args).as_mut_ptr();
    unsafe { MPI_Init(&mut argc as *mut _, &mut argv as *mut _) };
    Mpi
  }

  pub fn size(&self) -> usize {
    let mut size: c_int = 0;
    unsafe { MPI_Comm_size(MPI_COMM_WORLD, &mut size as *mut _) };
    size as usize
  }

  pub fn rank(&self) -> usize {
    let mut rank: c_int = 0;
    unsafe { MPI_Comm_rank(MPI_COMM_WORLD, &mut rank as *mut _) };
    rank as usize
  }

  pub fn send<T: MpiData>(&self, buf: &[T], dst: usize) {
    unsafe { MPI_Send(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, 0, MPI_COMM_WORLD) };
  }

  pub fn recv<T: MpiData>(&self, buf: &mut [T], src: usize) {
    let mut status: MPI_Status = Default::default();
    unsafe { MPI_Recv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src as c_int, 0, MPI_COMM_WORLD, &mut status as *mut _) };
  }

  pub fn barrier(&self) {
    unsafe { MPI_Barrier(MPI_COMM_WORLD) };
  }

  pub fn broadcast<T: MpiData>(&self, buf: &[T], root: usize) {
    unsafe { MPI_Bcast(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), root as c_int, MPI_COMM_WORLD) };
  }

  pub fn allreduce<T: MpiData, Op: MpiOp>(&self, sendbuf: &[T], recvbuf: &mut [T], _op: Op) {
    assert_eq!(sendbuf.len(), recvbuf.len());
    unsafe { MPI_Allreduce(sendbuf.as_ptr() as *const c_void, recvbuf.as_mut_ptr() as *mut c_void, sendbuf.len() as c_int, T::datatype(), Op::op(), MPI_COMM_WORLD) };
  }
}
