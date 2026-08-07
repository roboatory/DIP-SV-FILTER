#!/bin/bash
#
#SBATCH --job-name=DipSVFilter_Sniffles2
#
##### Account and Partition ####
#SBATCH --partition=interactive
#SBATCH --account=p_dsi_int
#SBATCH --qos=dsi_int
#
##### Resources Required ####
###Basic
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01-00:00:00
#
##### Environment Related ####
#SBATCH --export=ALL
#
##### Output and Error Information ####
#SBATCH --output=DipSVFilter.out
#SBATCH --error=DipSVFilter.err
#
##### Email Notification ####
#SBATCH --mail-user=yichen.liu.1@vanderbilt.edu
#SBATCH --mail-type=END,FAIL

ml miniconda3
source activate FPFilter

vcf=/data/maiziezhou_lab/Yichen/Projects/LRSV_Project/Vcf_Collection/NA24385_Pacbio_HiFI_alignment_based/NGMLR_newtools/Sniffles2_NGMLR.vcf 
prefix=Sniffles2
bam=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/Pacbio_data/NA24385/PacBio_CCS_15kb_20kb_chemistry2_reads_NA24385/NGMLR_aligned_bam/NA24385_aligned_by_ngmlr.sorted.bam
ref=/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa

python /data/maiziezhou_lab/Yichen/Projects/LRSV_Project/vcf_filter.py -v ${vcf} -o_dir . --dipcall -p ${prefix}

rm ${prefix}_DEL_noXY.vcf
rm ${prefix}_INS_noXY.vcf

python ../bin/vcf_preprocessing.py -i ${prefix}_DEL_INS_noXY.vcf -o ${prefix}.idnorm.vcf
rm ${prefix}_DEL_INS_noXY.vcf

python ../bin/construct_haplotypes.py -i ${prefix}.idnorm.vcf -o ./pseudo_haps -r ${ref}

python ../bin/kmer_filter_contig_pairs.py --fasta ./pseudo_haps/ --bam ${bam} --out-dir ./kmer_ranked/ --threads 32

python ../bin/align_cluster_reads_to_haps.py ./kmer_ranked/ --preset hifi

python ../bin/encode_pair_target_windows.py ./kmer_ranked/ --threads 32

python ../bin/classify_contig_pairs.py --model ../Pretrained_Models/Hifi_model.pt --cluster-dir ./kmer_ranked/ --overwrite

python ../bin/export_filtered_vcf.py \
    --cluster-dir ./kmer_ranked/ \
    --input-vcf ${prefix}.idnorm.vcf \
    --output-vcf ${prefix}.filtered.vcf  \
    --decision-tsv ${prefix}.filtered.tsv \
    --overwrite \
    --max-present-ins-allele-score 0.95 \
    --min-absent-del-allele-score 0.5

#clean up reads
rm ./kmer_ranked/*/cluster_reads.fasta
