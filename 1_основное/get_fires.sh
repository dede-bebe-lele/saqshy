#!/bin/sh
# Download NASA FIRMS VIIRS (S-NPP) yearly active-fire files for Kazakhstan, 2012-2025. No key needed.
mkdir -p data/firms
for y in $(seq 2012 2025); do
  curl -s -o data/firms/viirs_$y.csv "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/$y/viirs-snpp_${y}_Kazakhstan.csv"
done
