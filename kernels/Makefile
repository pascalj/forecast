kernels := $(wildcard *.cl)
objects := $(patsubst %.cl,%.aocx,$(kernels))

CL ?= aoc
AOCFLAGS ?= -g0


all: $(objects)

%.aocx: %.cl
	$(CL) $(AOCFLAGS)  $< -o $@

.PHONY: clean
clean:
	rm -f *.aoco *.aocr *.aocx
	rm -rf $(patsubst %.cl,%,$(kernels))
