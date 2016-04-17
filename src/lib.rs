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

pub struct MpiGroup {
  inner:    MPI_Group,
}

impl MpiGroup {
  pub fn ranges(&self, ranges: &[(usize, usize, usize)]) -> MpiGroup {
    let mut c_ranges: Vec<c_int> = Vec::with_capacity(3 * ranges.len());
    for i in 0 .. ranges.len() {
      c_ranges.push(ranges[i].0 as c_int);
      c_ranges.push((ranges[i].1 - 1) as c_int);
      c_ranges.push(ranges[i].2 as c_int);
    }
    let mut new_inner = MPI_Group(null_mut());
    unsafe { MPI_Group_range_incl(self.inner, ranges.len() as c_int, c_ranges.as_mut_ptr(), &mut new_inner as *mut _) };
    MpiGroup{inner: new_inner}
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

pub struct MpiStatus {
  pub src_rank: usize,
  pub tag:      c_int,
  pub error:    c_int,
}

impl MpiStatus {
  pub fn new(ffi_status: MPI_Status) -> MpiStatus {
    MpiStatus{
      src_rank: ffi_status.source as usize,
      tag:      ffi_status.tag,
      error:    ffi_status.error,
    }
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

  pub fn nonblocking_recv(buf: &mut [T], src_or_any: Option<usize>) -> Result<MpiRequest<T>, c_int> {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let mut request = MPI_Request(null_mut());
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, 0, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      _marker: PhantomData,
    })
  }
}

impl<T> MpiRequest<T> {
  pub fn query(&mut self) -> Result<Option<MpiStatus>, c_int> {
    // FIXME(20160416)
    unimplemented!();
  }

  pub fn wait(&mut self) -> Result<MpiStatus, c_int> {
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Wait(&mut self.inner as *mut _, &mut status as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiStatus{
      src_rank: status.source as usize,
      tag:      status.tag,
      error:    status.error,
    })
  }
}

pub struct MpiRequestList<T> {
  reqs: Vec<MPI_Request>,
  _marker:  PhantomData<T>,
}

impl<T> MpiRequestList<T> {
  pub fn new() -> MpiRequestList<T> {
    MpiRequestList{
      reqs: vec![],
      _marker:  PhantomData,
    }
  }

  pub fn clear(&mut self) {
    self.reqs.clear();
  }

  pub fn append(&mut self, request: MpiRequest<T>) {
    self.reqs.push(request.inner);
  }

  pub fn wait_all(&mut self) -> Result<(), c_int> {
    if self.reqs.is_empty() {
      return Ok(());
    }
    let code = unsafe { MPI_Waitall(self.reqs.len() as c_int, self.reqs.as_mut_ptr(), null_mut()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

#[derive(Clone, Copy)]
pub enum MpiWindowFenceFlag {
  Null,
}

#[derive(Clone, Copy)]
pub enum MpiWindowLockMode {
  Exclusive,
  Shared,
}

pub struct MpiWindow<T> {
  buf_addr: *mut T,
  buf_len:  usize,
  inner:    MPI_Win,
}

impl<T> Drop for MpiWindow<T> {
  fn drop(&mut self) {
    // FIXME(20160415): need to do a fence before freeing, otherwise it will
    // cause a seg fault!
    unsafe { MPI_Win_fence(0, self.inner) };
    unsafe { MPI_Win_free(&mut self.inner as *mut _) };
  }
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

  pub fn fence(&self, /*_flag: MpiWindowFenceFlag*/) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let mut assert = 0;
    let code = unsafe { MPI_Win_fence(assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn lock(&self, target_rank: usize, mode: MpiWindowLockMode) -> Result<(), c_int> {
    let lock_type = match mode {
      MpiWindowLockMode::Exclusive => MPI_LOCK_EXCLUSIVE,
      MpiWindowLockMode::Shared => MPI_LOCK_SHARED,
    };
    // FIXME(20160416): assert code.
    let assert = 0;
    let code = unsafe { MPI_Win_lock(lock_type, target_rank as c_int, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn unlock(&self, target_rank: usize) -> Result<(), c_int> {
    let code = unsafe { MPI_Win_unlock(target_rank as c_int, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn start(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let mut assert = 1; // MPI_MODE_NOCHECK
    let code = unsafe { MPI_Win_start(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn complete(&self) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let mut assert = 0;
    let code = unsafe { MPI_Win_complete(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn post(&self, group: &MpiGroup) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let mut assert = 0;
    let code = unsafe { MPI_Win_post(group.inner, assert, self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn wait(&self) -> Result<(), c_int> {
    // FIXME(20160416): assert code.
    let mut assert = 0;
    let code = unsafe { MPI_Win_wait(self.inner) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

impl<T> MpiWindow<T> where T: MpiData {
  pub unsafe fn rma_get(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize, _mpi: &Mpi) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as isize),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn rma_put(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize, _mpi: &Mpi) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Put(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as isize),
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

pub struct Mpi;

impl !Send for Mpi {}

impl Drop for Mpi {
  fn drop(&mut self) {
    // FIXME(20160417)
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
    //unsafe { MPI_Init(&mut argc as *mut _, &mut argv as *mut _) };
    let mut provided: c_int = 0;
    unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_MULTIPLE, &mut provided as *mut _) };
    assert_eq!(provided, MPI_THREAD_MULTIPLE);
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

  pub fn self_group(&self) -> MpiGroup {
    let mut inner = MPI_Group(null_mut());
    unsafe { MPI_Comm_group(MPI_Comm::SELF(), &mut inner as *mut _) };
    MpiGroup{inner: inner}
  }

  pub fn world_group(&self) -> MpiGroup {
    let mut inner = MPI_Group(null_mut());
    unsafe { MPI_Comm_group(MPI_Comm::WORLD(), &mut inner as *mut _) };
    MpiGroup{inner: inner}
  }

  pub fn send<T: MpiData>(&self, buf: &[T], dst: usize) {
    unsafe { MPI_Send(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, 0, MPI_Comm::WORLD()) };
  }

  pub fn recv<T: MpiData>(&self, buf: &mut [T], src: usize) {
    let mut status: MPI_Status = Default::default();
    unsafe { MPI_Recv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src as c_int, 0, MPI_Comm::WORLD(), &mut status as *mut _) };
  }

  pub fn blocking_send<T>(buf: &[T], dst: usize) -> Result<(), c_int> where T: MpiData {
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Send(buf.as_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), dst as c_int, 0, MPI_Comm::WORLD()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn blocking_recv<T>(buf: &mut [T], maybe_src: Option<usize>) -> Result<MpiStatus, c_int> where T: MpiData {
    let src_rank = maybe_src.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Recv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, 0, MPI_Comm::WORLD(), &mut status as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiStatus::new(status))
  }

  pub fn send_recv<T>(&self, send_buf: &[T], dst: usize, recv_buf: &mut [T], maybe_src: Option<usize>) -> Result<MpiStatus, c_int> where T: MpiData {
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Sendrecv(
        send_buf.as_ptr() as *const _, send_buf.len() as c_int, T::datatype(), dst as c_int, 0,
        recv_buf.as_mut_ptr() as *mut _, recv_buf.len() as c_int, T::datatype(), maybe_src.map_or(MPI_ANY_SOURCE, |r| r as c_int), 0,
        MPI_Comm::WORLD(), &mut status as *mut _,
    ) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiStatus::new(status))
  }

  pub fn barrier(&self) {
    unsafe { MPI_Barrier(MPI_Comm::WORLD()) };
  }

  pub fn broadcast<T: MpiData>(&self, buf: &mut [T], root: usize) {
    unsafe { MPI_Bcast(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), root as c_int, MPI_Comm::WORLD()) };
  }

  pub fn allreduce<T: MpiData, Op: MpiOp>(&self, sendbuf: &[T], recvbuf: &mut [T], _op: Op) {
    assert_eq!(sendbuf.len(), recvbuf.len());
    unsafe { MPI_Allreduce(sendbuf.as_ptr() as *const c_void, recvbuf.as_mut_ptr() as *mut c_void, sendbuf.len() as c_int, T::datatype(), Op::op(), MPI_Comm::WORLD()) };
  }
}
