# Protein Structure Prediction via MDS

This repository is a research project on protein structure prediction via multidimensional scaling (MDS).
It provides two solutions to solve multidimensional scaling, one is the closed-form solution
and the other is stress minimization using majorization (smacof).
If one is interested in the theory of MDS, please read the reference.


### Example

- Test on random data, random generate point coordinates and reconstruct the geometric equivariant coordinates.


    python test_random.py

- Test on protein structure prediction. Read CA coordinates from pdb and use MDS to reconstruct the geometric equivariant coordinates. 
When given ground-truth distance map, the closed-form solution achieves `RMSD=8.465e-6`.
The iterative method SMACOF improves RMSD further, i.e., `RMSD=8.450e-6` when initialized from the closed-form solution.


    python test_protein_prediction.py


### Citing this work

If you use the code, please cite

    @software {zz_psp_mds,
        author = {Zi Zhan},
        title = {Protein Structure Prediction via MDS},
        year = {2024},
        URL = {https://github.com/greatzizhan/mds_protein},
    }


### Reference:

[1] https://en.wikipedia.org/wiki/Multidimensional_scaling

[2] http://www.cs.umd.edu/~djacobs/CMSC828/MDSexplain.pdf

[3] https://cran.r-project.org/web/packages/smacof/smacof.pdf


### Acknowledgements:

Feel free to discuss if you have any questions.
