Rscript generate_data.R --outdir data_new --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist exponential,gaussian --noise gaussian,exponential --independent --size 10   
Rscript generate_data.R --outdir data_new --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist exponential,gaussian --noise gaussian,exponential --independent --size 100       
Rscript generate_data.R --outdir data_new --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist exponential,gaussian --noise gaussian,exponential --independent --size 1000       
Rscript generate_data.R --outdir data_new --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist exponential,gaussian --noise gaussian,exponential --independent --size 10000       


Rscript run_methods.R --ncl 3 --datadir data_new --outdir results_new
