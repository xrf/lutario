# Put your own customizations in this file
-include etc/local.mk

# List of programs to be built

.PHONY: all check
all: bin/main bin/main_qd
objs=$(main_objs) $(main_qd_objs) $(alloc_test) $(commutator_test_objs) $(irange_test) $(matrix_test)

check: bin/alloc_test bin/commutator_test bin/irange_test bin/matrix_test
	bin/alloc_test
	bin/commutator_test
	bin/irange_test
	bin/matrix_test

main_objs=src/main.o src/basis.o src/commutator.o src/imsrg.o src/math.o src/ode.o src/oper.o src/pairing_model.o src/quantum_dot.o src/str.o src/utility.o
bin/main: $(main_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(main_objs) -lblas -lsgode

main_qd_objs=src/main_qd.o src/basis.o src/commutator.o src/imsrg.o src/math.o src/ode.o src/oper.o src/pairing_model.o src/quantum_dot.o src/str.o src/utility.o
bin/main_qd: $(main_qd_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(main_qd_objs) -lblas -lsgode

alloc_test_objs=src/alloc_test.o
bin/alloc_test: $(alloc_test_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(alloc_test_objs)

commutator_test_objs=src/commutator_test.o src/basis.o src/commutator.o src/math.o src/oper.o src/pairing_model.o src/quantum_dot.o src/str.o src/utility.o
bin/commutator_test: $(commutator_test_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(commutator_test_objs) -lblas

irange_test_objs=src/irange_test.o
bin/irange_test: $(irange_test_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(irange_test_objs)

matrix_test_objs=src/matrix_test.o
bin/matrix_test: $(matrix_test_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(matrix_test_objs)

# Cleanup

.PHONY: clean
clean:
	rm -fr bin src/*.dep src/*.o src/*/*.o

# Object file generation

.SUFFIXES: .c .cpp .o
.cpp.o:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<
.c.o:
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ -c $<

# Dependency generation

.SUFFIXES: .c .cpp .dep
.cpp.dep:
	touch $@
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o /dev/null -M -MG -MP -MT $*.o -MF $@ $< || :
.c.dep:
	touch $@
	$(CC) $(CPPFLAGS) $(CFLAGS) -o /dev/null -M -MG -MP -MT $*.o -MF $@ $< || :
-include $(objs:.o=.dep)
