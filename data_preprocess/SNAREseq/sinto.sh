#!/bin/bash
sample=${1}
gunzip ${sample}_1.fastq.gz
gunzip ${sample}_2.fastq.gz
gunzip ${sample}_3.fastq.gz
sinto barcode -b 12 --barcode_fastq ${sample}_3.fastq \
    --read1 ${sample}_1.fastq \
    --read2 ${sample}_2.fastq
rm ${sample}_1.fastq
rm ${sample}_2.fastq
rm ${sample}_3.fastq
bwa mem -t 8 /mnt/raid62/Personal_data/zhangdan/reference/bwa/mm10.fa ${sample}_1.barcoded.fastq ${sample}_2.barcoded.fastq | samtools view -b - > ${sample}.bam
rm ${sample}_1.barcoded.fastq
rm ${sample}_2.barcoded.fastq
samtools sort -@ 15 ${sample}.bam -o ${sample}.sorted.bam
samtools index -@ 15 ${sample}.sorted.bam
sinto fragments -b ${sample}.sorted.bam -p 8 -f ${sample}.fragments.bed --barcode_regex "[^:]*"
rm ${sample}.bam

# index
sort -k1,1 -k2,2n ${sample}.fragments.bed > ${sample}.fragments.sort.bed
bgzip -@ 8 ${sample}.fragments.sort.bed
tabix -p bed ${sample}.fragments.sort.bed.gz