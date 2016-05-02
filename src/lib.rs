#![feature(optin_builtin_traits)]

extern crate libc;

use ffi::*;

use libc::{c_void, c_char, c_int};
use std::env;
use std::ffi::{CString, CStr};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::mem::{size_of};
use std::ptr::{null_mut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod ffi;

type AintTy = isize;

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

pub struct MpiMemory<T> {
  base: *mut T,
  len:  usize,
}

impl<T> Drop for MpiMemory<T> {
  fn drop(&mut self) {
    let code = unsafe { MPI_Free_mem(self.base as *mut _) };
    if code != 0 {
      panic!("MPI_Free_mem failed: {}", code);
    }
  }
}

impl<T> MpiMemory<T> {
  pub fn alloc_(len: usize) -> Result<MpiMemory<T>, c_int> {
    let mut base = null_mut();
    let code = unsafe { MPI_Alloc_mem(MPI_Aint((size_of::<T>() * len) as AintTy), MPI_Info::NULL(), &mut base as *mut *mut T as *mut *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiMemory{
      base: base,
      len:  len,
    })
  }
}

impl<T> AsRef<[T]> for MpiMemory<T> {
  fn as_ref(&self) -> &[T] {
    unsafe { from_raw_parts(self.base as *const T, self.len) }
  }
}

impl<T> AsMut<[T]> for MpiMemory<T> {
  fn as_mut(&mut self) -> &mut [T] {
    unsafe { from_raw_parts_mut(self.base, self.len) }
  }
}

pub struct MpiComm {
  inner:    MPI_Comm,
  predef:   bool,
}

impl Drop for MpiComm {
  fn drop(&mut self) {
    if !self.predef {
      // FIXME(20160419)
      //unsafe { MPI_Comm_disconnect(&mut self.inner as *mut _) };
    }
  }
}

impl MpiComm {
  pub fn self_() -> MpiComm {
    MpiComm{
      inner:    MPI_Comm::SELF(),
      predef:   true,
    }
  }

  pub fn world() -> MpiComm {
    MpiComm{
      inner:    MPI_Comm::WORLD(),
      predef:   true,
    }
  }

  pub fn accept(port_name: &CStr) -> Result<MpiComm, c_int> {
    let mut inner = unsafe { MPI_Comm::NULL() };
    let code = unsafe { MPI_Comm_accept(port_name.as_ptr(), MPI_Info::NULL(), 0, MPI_Comm::SELF(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiComm{
      inner:  inner,
      predef: false,
    })
  }

  pub fn connect(port_name: &CStr) -> Result<MpiComm, c_int> {
    let mut inner = unsafe { MPI_Comm::NULL() };
    let code = unsafe { MPI_Comm_connect(port_name.as_ptr(), MPI_Info::NULL(), 0, MPI_Comm::SELF(), &mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    //Ok(MpiComm{inner: inner})
    Ok(MpiComm{
      inner:  inner,
      predef: false,
    })
  }

  pub fn nonblocking_send<T>(&mut self, buf: &[T], dst: usize, tag: i32) -> Result<MpiRequest, c_int> where T: MpiData {
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Isend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
  }

  pub fn nonblocking_sync_send<T>(&mut self, buf: &[T], dst: usize, tag: i32) -> Result<MpiRequest, c_int> where T: MpiData {
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Issend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
  }

  pub fn nonblocking_probe(&mut self, src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<Option<MpiStatus>, c_int> {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut flag = 0;
    let mut status: MPI_Status = Default::default();
    let code = unsafe { MPI_Iprobe(src_rank, tag, self.inner, &mut flag as *mut _, &mut status as *mut _) };
    if code != 0 {
      return Err(code);
    }
    match flag {
      0 => Ok(None),
      1 => Ok(Some(MpiStatus::new(status))),
      _ => unreachable!(),
    }
  }

  pub fn nonblocking_recv<T>(&mut self, buf: &mut [T], src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, tag, self.inner, &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
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
    let mut new_inner = unsafe { MPI_Group::NULL() };
    unsafe { MPI_Group_range_incl(self.inner, ranges.len() as c_int, c_ranges.as_mut_ptr(), &mut new_inner as *mut _) };
    MpiGroup{inner: new_inner}
  }
}

pub struct MpiInfo {
  inner:    MPI_Info,
}

impl MpiInfo {
  pub fn null() -> MpiInfo {
    MpiInfo{
      inner:    unsafe { MPI_Info::NULL() },
    }
  }

  pub fn create(/*_mpi: &Mpi*/) -> Result<MpiInfo, c_int> {
    let mut inner = unsafe { MPI_Info::NULL() };
    let code = unsafe { MPI_Info_create(&mut inner as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiInfo{
      inner:    inner,
    })
  }

  pub fn set(&mut self, key: &CStr, value: &CStr) -> Result<(), c_int> {
    let code = unsafe { MPI_Info_set(self.inner, key.as_ptr(), value.as_ptr()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

/*impl Drop for MpiInfo {
  fn drop(&mut self) {
  }
}*/

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

pub struct MpiRequest {
  inner:    MPI_Request,
  //_marker:  PhantomData<T>
}

impl MpiRequest {
  pub fn nonblocking_send<T>(buf: &[T], dst: usize, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Isend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
  }

  pub fn nonblocking_sync_send<T>(buf: &[T], dst: usize, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Issend(buf.as_ptr() as *const c_void, buf.len() as c_int, T::datatype(), dst as c_int, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
  }

  pub fn nonblocking_recv<T>(buf: &mut [T], src_or_any: Option<usize>, tag_or_any: Option<i32>) -> Result<MpiRequest, c_int> where T: MpiData {
    let src_rank = src_or_any.map_or(MPI_ANY_SOURCE, |r| r as c_int);
    let tag = tag_or_any.unwrap_or(MPI_ANY_TAG);
    let mut request = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Irecv(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), src_rank, tag, MPI_Comm::WORLD(), &mut request as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{
      inner: request,
      //_marker: PhantomData,
    })
  }
}

impl MpiRequest {
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

pub struct MpiRequestList {
  reqs:   Vec<MPI_Request>,
  stats:  Vec<MPI_Status>,
  //_marker:  PhantomData<T>,
}

impl MpiRequestList {
  pub fn new() -> MpiRequestList {
    MpiRequestList{
      reqs:   vec![],
      stats:  vec![],
      //_marker:  PhantomData,
    }
  }

  pub fn clear(&mut self) {
    self.reqs.clear();
    self.stats.clear();
  }

  pub fn append(&mut self, request: MpiRequest) {
    self.reqs.push(request.inner);
    self.stats.push(MPI_Status::default());
  }

  pub fn wait_all(&mut self) -> Result<(), c_int> {
    if self.reqs.is_empty() {
      return Ok(());
    }
    let code = unsafe { MPI_Waitall(self.reqs.len() as c_int, self.reqs.as_mut_ptr(), self.stats.as_mut_ptr()) };
    if code != 0 {
      return Err(code);
    }
    self.reqs.clear();
    self.stats.clear();
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

pub struct MpiOwnedWindow<T, Storage> {
  buf:      Storage,
  inner:    MPI_Win,
  _marker:  PhantomData<T>,
}

impl<T, Storage> Drop for MpiOwnedWindow<T, Storage> {
  fn drop(&mut self) {
    // FIXME(20160415): need to do a fence before freeing, otherwise it will
    // cause a seg fault!
    unsafe { MPI_Win_fence(0, self.inner) };
    unsafe { MPI_Win_free(&mut self.inner as *mut _) };
  }
}

impl<T, Storage> MpiOwnedWindow<T, Storage> where Storage: AsMut<[T]> {
  pub fn create_(mut buf: Storage) -> Result<MpiOwnedWindow<T, Storage>, c_int> {
    let mut inner = unsafe { MPI_Win::NULL() };
    {
      let mut buf = buf.as_mut();
      let code = unsafe { MPI_Win_create(buf.as_mut_ptr() as *mut _, MPI_Aint((size_of::<T>() * buf.len()) as AintTy), size_of::<T>() as c_int, unsafe { MPI_Info::NULL() }, MPI_Comm::WORLD(), &mut inner as *mut _) };
      if code != 0 {
        return Err(code);
      }
    }
    Ok(MpiOwnedWindow{
      buf:      buf,
      inner:    inner,
      _marker:  PhantomData,
    })
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.buf.as_mut()
  }

  pub fn fence(&self, /*_flag: MpiOwnedWindowFenceFlag*/) -> Result<(), c_int> {
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

impl<T, Storage> MpiOwnedWindow<T, Storage> where T: MpiData, Storage: AsRef<[T]> {
  pub unsafe fn get_(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let buf_len = self.buf.as_ref().len();
    assert!(origin_len <= buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put_(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    let buf_len = self.buf.as_ref().len();
    assert!(origin_len <= buf_len);
    let code = MPI_Put(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put_accumulate_<Op>(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize, _op: Op) -> Result<(), c_int>
  where Op: MpiOp {
    let buf_len = self.buf.as_ref().len();
    assert!(origin_len <= buf_len);
    let code = MPI_Accumulate(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        Op::op(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }
}

pub struct MpiWindow<T> {
  buf_addr: *mut T,
  buf_len:  usize,
  inner:    MPI_Win,
}

unsafe impl<T> Send for MpiWindow<T> {}

impl<T> Drop for MpiWindow<T> {
  fn drop(&mut self) {
    // FIXME(20160415): need to do a fence before freeing, otherwise it will
    // cause a seg fault!
    unsafe { MPI_Win_fence(0, self.inner) };
    unsafe { MPI_Win_free(&mut self.inner as *mut _) };
  }
}

impl<T> MpiWindow<T> {
  pub unsafe fn create_(buf_addr: *mut T, buf_len: usize) -> Result<MpiWindow<T>, c_int> {
    let mut inner = MPI_Win::NULL();
    let code = MPI_Win_create(buf_addr as *mut _, MPI_Aint((size_of::<T>() * buf_len) as AintTy), size_of::<T>() as c_int, MPI_Info::NULL(), MPI_Comm::WORLD(), &mut inner as *mut _);
    if code != 0 {
      return Err(code);
    }
    Ok(MpiWindow{
      buf_addr: buf_addr,
      buf_len:  buf_len,
      inner:    inner,
    })
  }

  pub unsafe fn create(buf_addr: *mut T, buf_len: usize, _mpi: &Mpi) -> Result<MpiWindow<T>, c_int> {
    /*let info = match MpiInfo::create(_mpi) {
      Ok(info) => info,
      Err(e) => return Err(e),
    };*/
    let mut inner = MPI_Win::NULL();
    let code = MPI_Win_create(buf_addr as *mut _, MPI_Aint((size_of::<T>() * buf_len) as AintTy), size_of::<T>() as c_int, MPI_Info::NULL(), MPI_Comm::WORLD(), &mut inner as *mut _);
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
  pub unsafe fn get_(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put_(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Put(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn put_accumulate_<Op>(&self, origin_addr: *const T, origin_len: usize, target_rank: usize, target_offset: usize, _op: Op) -> Result<(), c_int>
  where Op: MpiOp {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Accumulate(
        origin_addr as *const _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
        origin_len as c_int,
        T::datatype(),
        Op::op(),
        self.inner,
    );
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub unsafe fn rma_get(&self, origin_addr: *mut T, origin_len: usize, target_rank: usize, target_offset: usize, _mpi: &Mpi) -> Result<(), c_int> {
    assert!(origin_len <= self.buf_len);
    let code = MPI_Get(
        origin_addr as *mut _,
        origin_len as c_int,
        T::datatype(),
        target_rank as c_int,
        MPI_Aint(target_offset as AintTy),
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
        MPI_Aint(target_offset as AintTy),
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

  pub fn new_serialized() -> Mpi {
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
    unsafe { MPI_Init_thread(&mut argc as *mut _, &mut argv as *mut _, MPI_THREAD_SERIALIZED, &mut provided as *mut _) };
    assert_eq!(provided, MPI_THREAD_SERIALIZED);
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

  pub fn nonblocking_broadcast_<T>(buf: &mut [T], root: usize) -> Result<MpiRequest, c_int>
  where T: MpiData {
    let mut req = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Ibcast(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), root as i32, MPI_Comm::WORLD(), &mut req as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{inner: req})
  }

  pub fn nonblocking_allreduce_<T, Op>(src_buf: &[T], dst_buf: &mut [T], _op: Op) -> Result<MpiRequest, c_int>
  where T: MpiData, Op: MpiOp {
    assert_eq!(src_buf.len(), dst_buf.len());
    let mut req = unsafe { MPI_Request::NULL() };
    let code = unsafe { MPI_Iallreduce(src_buf.as_ptr() as *const c_void, dst_buf.as_mut_ptr() as *mut c_void, src_buf.len() as c_int, T::datatype(), Op::op(), MPI_Comm::WORLD(), &mut req as *mut _) };
    if code != 0 {
      return Err(code);
    }
    Ok(MpiRequest{inner: req})
  }

  pub fn open_port_() -> Result<CString, c_int> {
    let mut port_buf: Vec<u8> = repeat(0).take(MPI_MAX_PORT_NAME as usize + 1).collect();
    let code = unsafe { MPI_Open_port(MPI_Info::NULL(), port_buf.as_mut_ptr() as *mut c_char) };
    if code != 0 {
      return Err(code);
    }
    let mut buf_len = 0;
    for i in 0 .. port_buf.len() {
      if port_buf[i] == 0 {
        buf_len = i;
        break;
      }
    }
    unsafe { port_buf.set_len(buf_len) };
    match CString::new(port_buf) {
      Ok(cstr) => Ok(cstr),
      Err(e) => panic!("failed to turn port into CString: {:?}", e),
    }
  }

  pub fn publish_service_(service_name: &CStr, global: bool, unique: bool, port: &CStr) -> Result<(), c_int> {
    let global_key_buf = b"ompi_global_scope".to_vec();
    let global_value_buf = match global {
      false => b"false".to_vec(),
      true  => b"true".to_vec(),
    };
    let unique_key_buf = b"ompi_unique".to_vec();
    let unique_value_buf = match unique {
      false => b"false".to_vec(),
      true  => b"true".to_vec(),
    };
    let global_key = CString::new(global_key_buf).unwrap();
    let global_value = CString::new(global_value_buf).unwrap();
    let unique_key = CString::new(unique_key_buf).unwrap();
    let unique_value = CString::new(unique_value_buf).unwrap();
    let mut info = MpiInfo::create().unwrap();
    info.set(&global_key, &global_value).unwrap();
    info.set(&unique_key, &unique_value).unwrap();
    let code = unsafe { MPI_Publish_name(service_name.as_ptr(), info.inner, port.as_ptr()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn lookup_service_(service_name: &CStr) -> Result<CString, c_int> {
    let order_key_buf = b"ompi_lookup_order".to_vec();
    let order_value_buf = b"local".to_vec();
    let order_key = CString::new(order_key_buf).unwrap();
    let order_value = CString::new(order_value_buf).unwrap();
    let mut info = MpiInfo::create().unwrap();
    info.set(&order_key, &order_value).unwrap();
    let mut port_buf: Vec<u8> = repeat(0).take(MPI_MAX_PORT_NAME as usize + 1).collect();
    let code = unsafe { MPI_Lookup_name(service_name.as_ptr(), info.inner, port_buf.as_mut_ptr() as *mut c_char) };
    if code != 0 {
      return Err(code);
    }
    let mut buf_len = 0;
    for i in 0 .. port_buf.len() {
      if port_buf[i] == 0 {
        buf_len = i;
        break;
      }
    }
    unsafe { port_buf.set_len(buf_len) };
    match CString::new(port_buf) {
      Ok(cstr) => Ok(cstr),
      Err(e) => panic!("failed to turn port into CString: {:?}", e),
    }
  }

  pub fn barrier_() -> Result<(), c_int> {
    let code = unsafe { MPI_Barrier(MPI_Comm::WORLD()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn broadcast_<T: MpiData>(buf: &mut [T], root: usize) -> Result<(), c_int> {
    let code = unsafe { MPI_Bcast(buf.as_mut_ptr() as *mut c_void, buf.len() as c_int, T::datatype(), root as c_int, MPI_Comm::WORLD()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn allreduce_<T: MpiData, Op: MpiOp>(sendbuf: &[T], recvbuf: &mut [T], _op: Op) -> Result<(), c_int> {
    assert_eq!(sendbuf.len(), recvbuf.len());
    let code = unsafe { MPI_Allreduce(sendbuf.as_ptr() as *const c_void, recvbuf.as_mut_ptr() as *mut c_void, sendbuf.len() as c_int, T::datatype(), Op::op(), MPI_Comm::WORLD()) };
    if code != 0 {
      return Err(code);
    }
    Ok(())
  }

  pub fn self_group(&self) -> MpiGroup {
    let mut inner = unsafe { MPI_Group::NULL() };
    unsafe { MPI_Comm_group(MPI_Comm::SELF(), &mut inner as *mut _) };
    MpiGroup{inner: inner}
  }

  pub fn world_group(&self) -> MpiGroup {
    let mut inner = unsafe { MPI_Group::NULL() };
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
