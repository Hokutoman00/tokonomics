# One-command entrypoints. `make demo` is the thing to run on any machine.
.PHONY: demo test dev project measured report figures arm-bench clean

# Full local experience: validate the pipeline on real x86 numbers, build the
# projection map, regenerate the report. No Arm, no compiler toolchain needed.
demo: dev project report
	@echo ""
	@echo "Done. See REPORT.md and figures/. results/measured/ is empty until"
	@echo "you run the Arm workflow (make arm-bench shows the command)."

test:
	pytest -q

dev:
	python -m tokonomics dev

project:
	python -m tokonomics project

# Merge CI driver JSONs (results/measured/bench_{off,on}.json) into measured
# economics. Run by the Arm workflow; needs the microkernel JSONs present.
measured:
	python -m tokonomics measured

report:
	python -m tokonomics report

figures: project
	@echo "figures regenerated under figures/"

arm-bench:
	@echo "Measured Arm numbers come from CI on the free ubuntu-24.04-arm runner."
	@echo "Fork the repo, then: Actions -> 'bench (Arm N2, measured)' -> Run workflow."
	@echo "Locally you can build the microkernel only on an Arm host:"
	@echo "  make -C bench/microkernel both && ./bench/microkernel/bench_on"

clean:
	rm -f bench/microkernel/bench_on bench/microkernel/bench_off bench/microkernel/bench_scalar
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
