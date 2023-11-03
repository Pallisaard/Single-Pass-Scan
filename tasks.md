# Tasks left to do for the project

## Implementation

- [x] Remove all unnecessary code from the project.
  - [x] stupid prints left by Rasmus.
  - [x] General cleanup.
- [x] Change benchmarks assumptions to fit with our golden algorithm.
  - [x] Set number of read/writes in GPU inclusive scan to 2.
- [ ] Change lookback scan to use multiple threads scan for better performance.
  - [x] Change lookback scan to use `WARP` number of threads for can.
  - [ ] Change lookback scan to use `BLOCKSIZE` number of threads for scan.
- [ ] Optimise lookback scan to use less global reads.
- [x] Implement templates for variadic types and implement tests.
  - [x] write templates for relevant functions.
  - [x] Write kernel tests for a `float32` version. **FLOATS ARE NON-COMMUTATIVE, SO WE CANT VALIDATE THEM WITH THE SCAN**
  - [x] Write kernel tests for a `(int32, int32, int32, int32)`.
- [ ] Follow up on code and remove unwanted kernels and functions.
- [ ] Optional: Implement fence free lookback version.

## Report

- [x] Create overleaf document.
- [x] Create benchmark code in python to compare performance of different implementations.
  - [x] Plot GB/s over N where N = 2^x for x in [10,30].
  - [x] For the best model, plot heatmap of GB/s for different combinations of Q, B and N:
    - [x] Q: [2, 4, 7, 8, 10, 13, 16, 20, 24, 30, 32, 40]
    - [x] B: [32, 64, 128, 256, 512, 1024]
    - [x] N: 2^x for x in [10, 30]
- [x] Create tables of the above for use in the appendix.
- [ ] Create code snippets for the report.
  - [x] Create code snippets for the general kernel implementation.
  - [ ] Write explanation of the how the kernel operates.
  - [ ] Create code snippet specifically for lookback scan.
  - [x] Write explanation of how the lookback scan works.
- [ ] Write the report.
  - [ ] Write the introduction.
  - [ ] Write the background.
  - [ ] Write the implementation.
  - [ ] Write the results.
  - [ ] Write the conclusion.
  - [ ] Write the appendix.
