# Build the int8 microkernel in two ablation variants from one source.
#   make ABLATION=on   -> Armv8.6 +i8mm (SMMLA)  binary  bench_on
#   make ABLATION=off  -> Armv8.2 +dotprod (SDOT) binary  bench_off
# On non-Arm hosts (x86 dev) the scalar reference path compiles for a smoke
# test; it is never used for timing (see driver/JSON kind="dev" upstream).

CC      ?= cc
ABLATION ?= off
COMMON   = -O3 -std=c11 -Wall -Wextra

ifeq ($(ABLATION),on)
  ARCH   = -march=armv8.6-a+i8mm -DUSE_I8MM
  LABEL  = on
  BIN    = bench_on
else
  ARCH   = -march=armv8.2-a+dotprod
  LABEL  = off
  BIN    = bench_off
endif

# Allow x86/dev hosts to compile the scalar path: if -march is rejected the
# caller can override ARCH= "" to fall through to the portable reference.
CFLAGS = $(COMMON) $(ARCH) -DABLATION_LABEL=\"$(LABEL)\"
SRC    = kernel.c bandwidth.c driver.c

.PHONY: all both clean
all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(BIN)

# Convenience: build both ablation variants in one go (used by CI).
both:
	$(MAKE) ABLATION=off
	$(MAKE) ABLATION=on

clean:
	rm -f bench_on bench_off
