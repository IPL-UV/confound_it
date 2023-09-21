Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist exponential --noise gaussian --independent --size 50
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist exponential --noise gaussian --independent --size 100
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist exponential --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist exponential --noise gaussian --independent --size 1000
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist exponential --noise gaussian --independent --size 5000

Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gamma --noise gaussian --independent --size 50
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gamma --noise gaussian --independent --size 100
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gamma --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gamma --noise gaussian --independent --size 1000
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gamma --noise gaussian --independent --size 5000

Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gaussian --noise gaussian --independent --size 50
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gaussian --noise gaussian --independent --size 100
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gaussian --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gaussian --noise gaussian --independent --size 1000
Rscript generate_data.R --rep 100 --latents 20 --proxy 50 --dist gaussian --noise gaussian --independent --size 5000




Rscript generate_data.R --rep 100 --latents 20 --proxy 100  --dist  gaussian --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 500  --dist  gaussian --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 1000 --dist  gaussian --noise gaussian --independent --size 500

Rscript generate_data.R --rep 100 --latents 20 --proxy 100  --dist  exponential --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 500  --dist  exponential --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 1000 --dist  exponential --noise gaussian --independent --size 500


Rscript generate_data.R --rep 100 --latents 20 --proxy 100  --dist  gamma --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 500  --dist  gamma --noise gaussian --independent --size 500
Rscript generate_data.R --rep 100 --latents 20 --proxy 1000 --dist  gamma --noise gaussian --independent --size 500


Rscript run_methods.R --ncl 5
