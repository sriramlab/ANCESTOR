# ANCESTOR
ANCESTOR requires Python module pymc.

To run ANCESTOR, type:
python ancestor.py input_filename output_filename

The input is a text file, where each row corresponds to one ancestor block in an individual. Each row should be tab-delimited and contains the following fields: chromosome, ancestry state 1, ancestry state 2, length in Morgan. Ancestry state 1 and 2 can be in arbitrary order. 

The output is a python dictionary and contains all the information from the MCMC run. Most of this is probably not needed for downstream analysis. The inferred genomic ancestries of both parents are printed at the end of the execution. 

To run the example file, type:
python ancestor.py test.txt test_out.dict

The example file takes ~5 mins to run. 
