Rscript generate_data.R  --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist gamma,uniform,exponentail,gaussian  --noise gaussian --independent --size 10       
Rscript generate_data.R  --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist gamma,uniform,exponentail,gaussian  --noise gaussian --independent --size 50       
Rscript generate_data.R  --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist gamma,uniform,exponentail,gaussian  --noise gaussian --independent --size 100
Rscript generate_data.R  --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist gamma,uniform,exponentail,gaussian  --noise gaussian --independent --size 500
Rscript generate_data.R  --noiseproxy 1 --rep 100 --latents 20 --proxy 1000 --dist gamma,uniform,exponentail,gaussian  --noise gaussian --independent --size 1000


Rscript run_methods.R --ncl 3 --oracle

Rscript evaluate.R --ncl 3 --oracle
