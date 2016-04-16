#![feature(optin_builtin_traits)]

extern crate libc;

use ffi::*;

use libc::{c_void, c_int};
use std::env;
use std::ffi::{CString};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ptr::{null_mut};

pub mod ffi;

pub trait MpiData {
  fn datatype() -> MPI_Datatype;
}

impl MpiData for u8 {
  fn datatype() -> MPI_Datatype {
    unsafe { MPI_Datatype::BYTE() }
  }
}

impl MpiData for u64 {
  fn datatype() -> MPI_Datatype {
    unsafe { MPI_Datatype::UNSIGNED_LONG_LONG() }
  }
}

impl MpiData for f32 {
  fn datatype() -> MPI_Datatype {
    unsafe { MPI_Datatype::FLOAT() }
  }
}

impl MpiData for f64 {
  fn datatype() -> MPI_Datatype {
    unsafe { MPI_Datatype::DOUBLE() }
  }
}

pub trait MpiOp {
  fn op() -> MPI_Op;
}

pub struct MpiSumOp;

impl MpiOp for MpiSumOp {
  fn op() -> MPI_Op {
    unsafe { MPI_Op::SUM() }
  }
}

pub struct MpiInfo {
  inner:    MPI_Info,
}

impl MpiInfo {
  pub unsafe fn create(_mpi: &Mpi) -> Result<MpiInfo, c_int> {
    let mut inner = MPI_Info(null_mut());
    let code = MPI_Info_create(&mut inner as *mut _);
    if code != 0 {
      return Err(code);
    }
    Ok(MpiInfo{
      inner:    inner,
    })
  }
}

impl Drop for MpiInfo {
  fn drop(&mut self) {
  }
}

pub struct MpiRequest<T> {
  inner:    MPI_Request,
  _marker:  PhantomData<T>
}

impl<T> MpiRequest<T> where T: MpiData {
  pub fn nonblocking_send(buf: &[T], dst: usize) -> Result<MpiRequest<T>, c_int> {
    let mut request = MPI_Request(null_mut());
    let code = unsafe { MPI_Isend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, 0, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      _marker: PhantomData,
    })
  }

  pub fn nonblocking_recv(buf: &mut [T], src: usize) -> Result<MpiRequest<T>, c_int> {
    let mut request = MPI_Request(null_mut());
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src as c_int, 0, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      _marker: PhantomData,
    })
  }

  pub fn blocking_wait(&self) -> Result<(), c_int> {
    // FIXME(20160415)
    unimplemented!();
  }
}

#[derive(Clone, Copy)]
pub enum MpiWindowFenceFlag {
}

pub struct MpiWindow<T> {
  buf_addr: *mut T,
  buf_len:  usize,
  inner:    MPI_Win,
}

impl<T> MpiWindow<T> {
  pub unsafe fn create(buf_addr: *mut T, buf_len: usize, _mpi: &Mpi) -> Result<MpiWindow<T>, c_int> {
    /*let info = match MpiInfo::create(_mpi) {
      Ok(info) => info,
      Err(e) => return Err(e),
    };*/
    let mut inner = MPI_Win(null_mut());
    let code = MPI_Win_create(buf_addr as *mut _, MPI_Aint((size_of::<T>() * buf_len) as isize), size_of::<T>() as c_int, MPI_Info::NULL(), MPI_Comm::WORLD(), &mut inner as *mut _);
    if code != 0 {
      return Err(code);
    }
    Ok(MpiWindow{
      buf_addr: buf_addr,
      buf_len:  buf_len,
      inner:    inner,
    })
  }

  pub fn fence(&self, flag: MpiWindowFenceFlag) -> Result<(), c_int> {
    // FIXME(20160415): fence flag.
    unimplemented!();
    let code = unsafe { MPI_Win_fence(0, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

impl<T> MpiWindow<T> where T: MpiData {
  pub unsafe fn rma_get(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, _mpi: &Mpi) -> Result<(), c_int> {
    assert_eq!(origin_len, self.buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(0),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

impl<T> Drop for MpiWindow<T> {
  fn drop(&mut self) {
    // FIXME(20160412)
    //unimplemented!();
  }
}

pub struct Mpi;

impl !Send for Mpi {}

impl Drop for Mpi {
  fn drop(&mut self) {
    //unsafe { MPI_Finalize() };
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
    unsafe { MPI_Comm_size(MPI_Comm::WORLD(), &mut size as *mut _) };
    size as usize
  }

  pub fn rank(&self) -> usize {
    let mut rank: c_int = 0;
    unsafe { MPI_Comm_rank(MPI_Comm::WORLD(), &mut rank as *mut _) };
    rank as usize
  }

  pub fn send<T: MpiData>(&self, buf: &[T], dst: usize) {
    unsafe { MPI_Send(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, 0, MPI_Comm::WORLD()) };
  }

  pub fn recv<T: MpiData>(&self, buf: &mut [T], src: usize) {
    let mut status: MPI_Status = Default::default();
    unsafe { MPI_Recv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src as c_int, 0, MPI_Comm::WORLD(), &mut status as *mut _) };
  }

  pub fn barrier(&self) {
    unsafe { MPI_Barrier(MPI_Comm::WORLD()) };
  }

  pub fn broadcast<T: MpiData>(&self, buf: &[T], root: usize) {
    unsafe { MPI_Bcast(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), root as c_int, MPI_Comm::WORLD()) };
  }

  pub fn allreduce<T: MpiData, Op: MpiOp>(&self, sendbuf: &[T], recvbuf: &mut [T], _op: Op) {
    assert_eq!(sendbuf.len(), recvbuf.len());
    unsafe { MPI_Allreduce(sendbuf.as_ptr() as *const c_void, recvbuf.as_mut_ptr() as *mut c_void, sendbuf.len() as c_int, T::datatype(), Op::op(), MPI_Comm::WORLD()) };
  }
}
