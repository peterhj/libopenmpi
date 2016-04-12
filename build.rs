fn main() {
  //println!("cargo:rustc-link-search=dylib=/usr/lib/openmpi/lib");
  //println!("cargo:rustc-link-search=dylib=/opt/openmpi/lib");
  println!("cargo:rustc-flags=-L /opt/openmpi/lib -l mpi -l hwloc");
}
