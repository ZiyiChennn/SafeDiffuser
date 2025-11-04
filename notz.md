When training a diffusion model, you need to comment out **b\_min** in diffusion.py.    locomotion:diffuser/models/diffusion.py     
maze:      diffuser/models/diffusion.py     line 1321 \& 1330-1331 \& line 1337-1338
kuka:      denoising\_diffusion\_pytorch/denoising\_diffusion\_pytorch.py



When using invariance or invariance\_cpx, a QP problem occurs, such as "Cannot perform LU factorization on Q. Please make sure that your Q matrix is PSD and has a non-zero diagonal." This is because the PyTorch version is too low. You need to upgrade to a version of PyTorch that has '**torch.linalg.lu\_solve**'.







When planning the trajectory, an image interface problem occurs. In the terminal, just input **export QT\_QPA\_PLATFORM=offscreen**.





All comments with --Ziyi-- mean the change is from me, not WeiXiao.







ImportError: /usr/lib/x86\_64-linux-gnu/libstdc++.so.6: version `GLIBCXX\_3.4.29' not found (required by /opt/conda/envs/safediffuser\_g10/lib/python3.8/site-packages/scipy/\_lib/\_uarray/\_uarray.cpython-38-x86\_64-linux-gnu.so)

Solution: export LD\_PRELOAD=$CONDA\_PREFIX/lib/libstdc++.so.6







Wenn planning the trajectory, need to change the locomotion: diffuser/models/diffusion.py    					line 190

                                                             scripts/plan\_guided.py          				line 130

                                                 maze:       diffuser/models/diffusion.py    					line 1210

                                                             scripts/plan\_maze2d.py         					line 61

                                      					 kuka:	     denoising\_diffusion\_pytorch/denoising\_diffusion\_pytorch.py         line 34  \&  line 428








