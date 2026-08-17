CXX ?= clang++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic
SUITESPARSE_PREFIX ?= /usr/local
CPPFLAGS += -Iinclude -I$(SUITESPARSE_PREFIX)/include/suitesparse
LDFLAGS += -L$(SUITESPARSE_PREFIX)/lib
LDLIBS += -lspqr -lcholmod -lsuitesparseconfig -framework Accelerate
OSQP_PREFIX ?= /usr/local/opt/osqp
OSQP_LDLIBS = -L$(OSQP_PREFIX)/lib -Wl,-rpath,$(OSQP_PREFIX)/lib -losqp
# Location of libhighs.  By default it is discovered from the Python
# environment that has highspy installed: point PYTHON at your virtualenv
# interpreter (e.g. `make PYTHON=.venv/bin/python`), or set HIGHS_LIBDIR
# directly to the directory holding libhighs.
PYTHON ?= python3
HIGHS_LIBDIR ?= $(shell $(PYTHON) -c "import highspy, os; print(os.path.dirname(highspy.__file__))" 2>/dev/null)
HIGHS_LDLIBS = -L$(HIGHS_LIBDIR) -Wl,-rpath,$(HIGHS_LIBDIR) -lhighs.1

.PHONY: all clean revised-probe revised-column-probe revised-svd-probe \
	revised-walker kernel-projection-probe triangular-projection-probe \
	quotient-spqr-kernel quotient-cholmod-kernel

all: build/fill_probe build/selector_probe build/simplex_probe build/symbolic_probe \
     build/verify_face_solver build/verify_gram_solver \
     build/verify_bound_core_solver build/verify_walker

# Standalone component probes remain outside the production answer path.
revised-probe: build/revised_basis_probe

revised-column-probe: build/revised_column_probe

revised-svd-probe: build/incremental_svd_probe

kernel-projection-probe: build/kernel_projection_probe

triangular-projection-probe: build/triangular_projection_probe

quotient-spqr-kernel: build/libquotient_spqr_kernel.dylib

quotient-cholmod-kernel: build/libquotient_cholmod_kernel.dylib

build/libquotient_spqr_kernel.dylib: revised/quotient_spqr_kernel.cpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -dynamiclib $< \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/libquotient_cholmod_kernel.dylib: \
    revised/quotient_cholmod_kernel.cpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -dynamiclib $< \
	  $(LDFLAGS) $(LDLIBS) -o $@

revised-walker: build/verify_revised_walker

build/revised_basis_probe: revised/revised_basis_probe.cpp \
                           revised/revised_basis_solver.cpp \
                           revised/revised_basis_solver.hpp \
                           include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised $(CXXFLAGS) \
	  revised/revised_basis_probe.cpp revised/revised_basis_solver.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/revised_column_probe: revised/revised_column_probe.cpp \
                            revised/revised_column_solver.cpp \
                            revised/revised_column_solver.hpp \
                            revised/revised_basis_solver.cpp \
                            revised/revised_basis_solver.hpp \
                            src/face_solver.cpp include/face_solver.hpp \
                            include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised $(CXXFLAGS) \
	  revised/revised_column_probe.cpp revised/revised_column_solver.cpp \
	  revised/revised_basis_solver.cpp src/face_solver.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/incremental_svd_probe: revised/incremental_svd_probe.cpp \
                             include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised $(CXXFLAGS) $< \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/kernel_projection_probe: revised/kernel_projection_probe.cpp \
                               revised/kernel_projection.cpp \
                               revised/kernel_projection.hpp \
                               include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised $(CXXFLAGS) \
	  revised/kernel_projection_probe.cpp revised/kernel_projection.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/triangular_projection_probe: revised/triangular_projection_probe.cpp \
                                   revised/triangular_projection.cpp \
                                   revised/triangular_projection.hpp \
                                   revised/kernel_projection.cpp \
                                   revised/kernel_projection.hpp \
                                   src/face_solver.cpp \
                                   include/face_solver.hpp \
                                   include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised $(CXXFLAGS) \
	  revised/triangular_projection_probe.cpp \
	  revised/triangular_projection.cpp revised/kernel_projection.cpp \
	  src/face_solver.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/verify_revised_walker: build/verify_walker | build
	cp $< $@

build:
	mkdir -p $@

build/fill_probe: bench/fill_probe.cpp include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/selector_probe: bench/selector_probe.cpp include/fixture.hpp \
                      include/highs_c_api_minimal.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) \
	  $(HIGHS_LDLIBS) -o $@

build/simplex_probe: bench/simplex_probe.cpp include/fixture.hpp \
                     include/highs_c_api_minimal.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) \
	  $(HIGHS_LDLIBS) -o $@

build/symbolic_probe: bench/symbolic_probe.cpp include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/verify_face_solver: tests/verify_face_solver.cpp src/face_solver.cpp \
                          include/face_solver.hpp include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/verify_face_solver.cpp \
	  src/face_solver.cpp $(LDFLAGS) $(LDLIBS) -o $@

build/verify_gram_solver: tests/verify_gram_solver.cpp \
                          src/gram_face_solver.cpp src/face_solver.cpp \
                          include/gram_face_solver.hpp include/face_solver.hpp \
                          include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/verify_gram_solver.cpp \
	  src/gram_face_solver.cpp src/face_solver.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/verify_bound_core_solver: tests/verify_bound_core_solver.cpp \
                                src/bound_core_face_solver.cpp \
                                include/bound_core_face_solver.hpp \
                                include/face_solver.hpp include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/verify_bound_core_solver.cpp \
	  src/bound_core_face_solver.cpp src/face_solver.cpp \
	  $(LDFLAGS) $(LDLIBS) -o $@

build/verify_walker: tests/verify_walker.cpp src/walker.cpp src/dense_nnls.cpp \
                     include/dense_nnls.hpp src/face_solver.cpp \
                     src/gram_face_solver.cpp \
					 src/bound_core_face_solver.cpp \
                     revised/revised_column_solver.cpp \
                     revised/revised_column_solver.hpp \
                     revised/augmented_kkt_basis.cpp \
                     revised/augmented_kkt_basis.hpp \
                     revised/maintained_deficient_qr_solver.cpp \
                     revised/maintained_deficient_qr_solver.hpp \
                     revised/maintained_rowspace_solver.cpp \
                     revised/maintained_rowspace_solver.hpp \
                     revised/maintained_svd_face_solver.cpp \
                     revised/maintained_svd_face_solver.hpp \
                     revised/kernel_projection.cpp \
                     revised/kernel_projection.hpp \
	                 revised/triangular_projection.cpp \
	                 revised/triangular_projection.hpp \
                     revised/revised_basis_solver.hpp \
                     include/walker.hpp include/face_solver.hpp \
                     include/gram_face_solver.hpp include/fixture.hpp | build
	$(CXX) $(CPPFLAGS) -Irevised -I$(OSQP_PREFIX)/include/osqp \
	  -DTWALKER_ENABLE_REVISED_COLUMN $(CXXFLAGS) \
	  tests/verify_walker.cpp \
	  src/walker.cpp src/dense_nnls.cpp src/face_solver.cpp src/gram_face_solver.cpp \
	  src/bound_core_face_solver.cpp \
	  revised/revised_column_solver.cpp revised/augmented_kkt_basis.cpp \
	  revised/maintained_deficient_qr_solver.cpp \
	  revised/maintained_rowspace_solver.cpp \
	  revised/maintained_svd_face_solver.cpp \
	  revised/kernel_projection.cpp revised/triangular_projection.cpp \
	  $(LDFLAGS) $(LDLIBS) \
	  $(HIGHS_LDLIBS) $(OSQP_LDLIBS) -o $@

clean:
	rm -f build/fill_probe build/selector_probe build/simplex_probe \
	  build/symbolic_probe \
	  build/verify_face_solver build/verify_gram_solver \
	  build/verify_bound_core_solver build/verify_walker \
	  build/revised_basis_probe build/revised_column_probe \
	  build/incremental_svd_probe \
	  build/kernel_projection_probe \
	  build/triangular_projection_probe \
	  build/libquotient_spqr_kernel.dylib \
	  build/libquotient_cholmod_kernel.dylib \
	  build/verify_revised_walker
