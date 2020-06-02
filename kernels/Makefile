kernels := $(wildcard *.cl)
objects := $(patsubst %.cl,%.aocx,$(kernels))
variants := vector_triad_n1.aocx vector_triad_n2.aocx vector_triad_n4.aocx
objects := $(objects) $(variants)

CL ?= aoc
AOCFLAGS ?= -g0


all: $(objects)

simulator: report

report: AOCFLAGS += -march=simulator -report -rtl
report: $(patsubst %.aocx,%.aocr,$(objects))


emulator: AOCFLAGS += -march=emulator
emulator: $(objects)

vector_triad_n%: NUM_KERNELS = $(shell echo $@ | sed 's/[^0-9]*//g')
vector_triad_n%: AOCFLAGS += -DNUM_KERNELS=$(NUM_KERNELS)
vector_triad_n%:
	$(CL) $(AOCFLAGS) vector_triad.cl -o $@

%.aocx: %.cl
	$(CL) $(AOCFLAGS)  $< -o $@

%.aocr: %.cl
	$(CL) $(AOCFLAGS)  $< -o $@

.PHONY: clean
clean:
	rm -f *.aoco *.aocr *.aocx
	rm -rf $(patsubst %.cl,%,$(kernels))

dist-clean: clean
	find . -mindepth 1 -maxdepth 1 -type d -exec rm -r {} +
