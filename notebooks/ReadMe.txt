# notebooks directory

For storing jupyter notebooks useful in setting up and analyzing simulations.  These notebooks calculate elastic and interaction quantities for setting up potential cg shell simulations, and perform analysis of already completed simulation runs.

List of notebooks:
2DAdhesion.ipynb
2DFlat.ipynb
2DPoisson.ipynb


Data structure of simulations produced by these notebooks:

cgshells (PROJECT_ROOT is variable that saves absolute path to here)
├── data
    ├── 2d
    │   ├── mesh_characteristics
            └── wx-56.00-t0-1.00-Nbeads-180
                ├── interactions
                        1patch
                        patchy
                            sigma-epsilon-soft
                                correct (correct binding orientation e.g 0-1 0-1)
                                    ysep
                                incorrect1 (e.g. 0-1 1-0)
                                    ysep
                                incorrect2 (e.g 1-0 0-1)
                                    ysep
                    elasticity
                        └── kvkh-1.00-kckh-1.00 (spring constant ratios)
                                stretch-x (poisson ratio test)
                                    kh
                                        force
                                stretch-y
                                    kh
                                        force
                            └── flat (flattening energy)
                                    r0-cg-14 (minimization method-etolerance e.g 14 = 1e-14)
                                        kh
            
    └── 3d
