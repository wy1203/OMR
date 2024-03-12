# Optical Music Recognition
CS4701 Group42

## Introduction to the GrandStaff Dataset 
- The dataset we use to train our model is from the paper [$\textit{End-to-end Optical Music Recognition for Pianoform Sheet Music}$](https://link.springer.com/article/10.1007/s10032-023-00432-z), written by Ríos-Vila, A., Rizo, D., Iñesta, J.M. et al.
- The history of OMR process includes the following stages
  - Stage 1: \
    There is a first set in which the basic symbols such as note heads, beams, or accidentals (usually referred to as “primitives”) are detected
  - Stage 2: \
    heuristic strategies based on hand-crafted rules, such as the ones we read from Orchestra Project
- Scheme: `**bekern`, i.e. basic extended kern
  - based on [**`kern` scheme](https://www.humdrum.org/guide/ch02/)
  - A `kern` file is a sequence of lines
  - Details please refer to [Hundrum Tool Kit](https://www.humdrum.org/rep/kern/)
- Load the dataset
    ```
    cd $PATH_TO_OMR/OMR
    python extract_dataset.py
    ```

## Useful Links for Report
- [Robust and Adaptive OMR System Including Fuzzy Modeling, Fusion of Musical Rules, and Possible Error Detection](https://doi.org/10.1155/2007/81541)
- 
