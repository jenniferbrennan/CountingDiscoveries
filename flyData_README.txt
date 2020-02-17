README for fly data
-------------------

The file flyData.txt contains data from Hao et. al. (2008), which we used in our experiments. The email chain below describes the source of the data. The following code snippet can be used to load the data into Python:

	# load raw data
	X = []
	k=0
	l=0
	# Here we extract two Z-scores per gene (there are 13071 genes)
	with open('flyData.txt', newline='') as csvfile:
	    reader = csv.DictReader(csvfile,delimiter='\t')
	    print(reader.fieldnames)
	    for row in reader:
	        if row['target_name']=='empty':
	            k +=1
	        else:
	            l += 1
	            first_Z_score = float(row['1 z w/o controls'])
	            second_Z_score = float(row['2 z w/o controls'])
	            X.append([first_Z_score, second_Z_score ])
	print(k,l)       # Should be 753 13071
	X = -np.array(X) # flip sign so large observations are discoveries
	print(X.shape)   # Should be (13071, 2)

	mu_hat = np.mean(X, axis=1) # average the 2 Z-scores together
	sigma_sq = 1.               # variance of Z-score, by definition
	gamma = sigma_sq/2          # variance of average of both

After loading the data in this way, the vector muHat will be the vector of test statistics, where each is assumed to be normal with mean zero and variance gamma. We found that the variance was not well described by the theoretical value gamma, and instead elected to fit the variance (as described in our paper).



#-------------------------------------------------------------------------------
#- original emails.
#--------------------------------------------------------------------------------

Michael Newton <newton@biostat.wisc.edu>
AttachmentsApr 23 (5 days ago)

to Robert, Paul, Robert, Kwang-Sung 
Hi Rob,

Attached please find the original data from Lin Hao back in 2008, which were summarized in the Nature paper.
The tab-delmited text file has 13824 rows, one per basic measurement. Those rows for which the `target_name' column is not `empty' correspond to Drosophila genes (13071).   Columns 4 and 10 (having a `z' in the title) correspond to z-scores from two replicates of the primary genome-wide screen. Other columns indicate plate/well info as well as raw data.  Genes selected for a secondary screen were extreme on both replicates, as I recall.  I can organize the data I have from the second screen also, but that's not part of this first file. 
Apologies again for the delay. Let me know if I can clarify aspects of these data.

-Michael N.

--------------------------------------------------------------------------------
Kwang-Sung Jun <deltakam@cs.wisc.edu>
Apr 24 (4 days ago)

to Michael, Robert, Paul, Robert 
Dear Prof. Newton,

I am a PhD student in CS department working with Jerry Zhu. I have been working closely with Rob on an adaptive method for doing the microwell experiment. First of all, I'd like to thank you for sharing the experiment data and we are very excited to run our algorithm on it.

Our algorithm works best with sub-Gaussian, so we wanted to transform the data into a nicer domain such as logarithm. Then, we think it might be okay to assume that the log of measurement noise follows Gaussian for our purpose.

So, here comes a question.
1. Is the log-normal assumption something you would do? Are there studies on modeling the measurement noise in this type of data?
2. Can you suggest some other transformations that makes the measurement similar to Gaussian?

Thanks!

Best,
Kwang-Sung Jun

--------------------------------------------------------------------------------

Michael Newton
Apr 27 (1 day ago)

to Kwang-Sung, Robert, Paul, Robert 
Hi Kwang-Sung,

The measurements were processed to eliminate plate effects (I think 384 wells per plate, log scale data centered and scaled by standard deviation to produce Z scores).
The file I sent carries Z-scores for two replicate screens.

-Michael N.

--------------------------------------------------------------------------------

Kwang-Sung Jun <deltakam@cs.wisc.edu>
Apr 27 (1 day ago)

to Michael, Robert, Paul, Robert 
Cool. Could you please provide an explanation on eliminating plate effects, or a reference to it? It's important for us to justify our simulation. Thanks!

Best,
Kwang-Sung Jun

--------------------------------------------------------------------------------

Michael Newton
Apr 27 (1 day ago)

to Kwang-Sung, Robert, Paul, Robert 
Hi Kwang-Sung,
Well plates are a basic unit of the experiment; for reasons that have nothing to do with genetic or RNAi effects, it is always possible that measurements on the same plate may be more similar than measurements on different plates, owing to experimental exigencies.  Such blocking or batch effects need to be accommodated, and the simplest scheme for such is to center and rescale the data  per plate.  As to the raw optical measures, these are I think from photon counts in a photo-multiplier tube, and such data routinely exhibits increased variation with increased mean.  Taking log counters that effect.  I recommend you take a look at the raw data and examine these variation issues.  You might also check the classic reference by Box Hunter and Hunter (statistics for experimenters), where such issues are discussed.
thanks
-Michael N.
