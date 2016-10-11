# Put your own customizations in this file
-include etc/local.mk

# List of programs to be built

.PHONY: all
all: bin/main
objs=$(main_objs)

main_objs=src/main.o src/basis.o src/commutator.o src/oper.o src/pairing_model.o
bin/main: $(main_objs)
	mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(main_objs) -lblas

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
