##################################### 10X PBMC call peaks by ARCHR ###############################
##################################### step1: create project #######################
# 10X PBMC reference is : hg38
suppressMessages(library(ArchR))
suppressMessages(library(patchwork))
suppressMessages(library(BSgenome))
suppressMessages(library(BSgenome.Hsapiens.UCSC.hg38))
suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))
suppressMessages(library(cowplot))
suppressMessages(library(ggrepel))

# set parameters
set.seed(42)
addArchRThreads(threads = parallel::detectCores() - 2)
addArchRGenome("hg38")

## create Arrow files
fragment_files = "/mnt/data3/zhangdan/multiomics/D5_ASAP/processed_dataset/GSM4732109_CD28_CD3_control_ASAP_fragments.tsv.gz"
fragment_names = gsub("_ASAP_fragments.tsv.gz", "", basename(fragment_files))
inputFiles <- fragment_files
names(inputFiles) = fragment_names

minTSS <- 4
minFrags <- 500

setwd("/mnt/data3/zhangdan/multiomics/D5_ASAP/peak_V3/script")
ArrowFiles <- createArrowFiles(
  inputFiles = inputFiles,
  sampleNames = names(inputFiles),
  outputNames = names(inputFiles),
  minTSS = minTSS, 
  minFrags = minFrags, 
  QCDir = "../data/QualityControl",
  addTileMat = TRUE,
  addGeneScoreMat = TRUE
)

# visual quality control
## plot quality control
plotlist <- lapply(names(inputFiles), function(sample){
  input_filename <- sprintf("../data/QualityControl/%s/%s-Pre-Filter-Metadata.rds", sample, sample)
  
  if(file.exists(input_filename)){
    Metadata <- readRDS(input_filename)
    
    ggtitle <- sprintf("%s",
                       paste0(sample, "\nnCells Pass Filter = ", sum(Metadata$Keep))
    )
    
    gg <- ggPoint(
      x = pmin(log10(Metadata$nFrags), 5) + rnorm(length(Metadata$nFrags), sd = 0.00001),
      y = Metadata$TSSEnrichment + rnorm(length(Metadata$nFrags), sd = 0.00001), 
      colorDensity = TRUE,
      xlim = c(2.5, 5),
      ylim = c(0, max(Metadata$TSSEnrichment) * 1.05),
      baseSize = 6,
      continuousSet = "sambaNight",
      xlabel = "Log 10 (Unique Fragments)",
      ylabel = "TSS Enrichment",
      title = ggtitle,
      rastr = TRUE) + 
      geom_hline(yintercept=minTSS, lty = "dashed", size = 0.25) +
      geom_vline(xintercept=log10(minFrags), lty = "dashed", size = 0.25) +
      theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, unit = "in"))
    
    return(gg)  
  }
})

p <- patchwork::wrap_plots(plotlist, ncol = 1)

options(repr.plot.width = 28, repr.height = 16)

## Creating an ArchRProject
proj <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "../data/VisiumHeart",
  showLogo = FALSE,
  copyArrows = TRUE #This is recommened so that you maintain an unaltered copy for later usage.
)

getAvailableMatrices(proj)

# Inferring scATAC-seq Doublets with ArchR
#proj <- addDoubletScores(
#  input = proj,
#  k = 10, #Refers to how many cells near a "pseudo-doublet" to count.
#  knnMethod = "UMAP", #Refers to the embedding to use for nearest neighbor search with doublet projection.
#  LSIMethod = 1,
#  outDir = "../data/DoubletScores"
#)

## remove doublets
#proj <- filterDoublets(proj)

## Save data
saveArchRProject(ArchRProj = proj, 
                 load = FALSE)

###################################### step2: call peak by macs2 #############################

proj <- loadArchRProject("../data/VisiumHeart", showLogo = FALSE)
peakcells <- rownames(proj@cellColData)
cells <- read.csv("/mnt/data3/zhangdan/multiomics/D5_ASAP/asap_cell.txt", header = FALSE)
cells$V1 <- gsub("Control#", "", cells$V1)
peakcells <- gsub("GSM4732109_CD28_CD3_control#", "", peakcells)
print(paste0("########################total is 4502,common cells is" ,
             length(intersect(peakcells, cells$V1))))

proj <- addGroupCoverages(ArchRProj = proj, groupBy = "Sample")

pathToMacs2 <- "/mnt/data1/zhangdan/software/miniconda3/condainstall/bin/macs2"
proj <- addReproduciblePeakSet(
  ArchRProj = proj, 
  groupBy = "Sample", 
  pathToMacs2 = pathToMacs2
)

proj <- addPeakMatrix(proj)

## save peak matrix
peakMatrix <- getMatrixFromProject(proj,
                                   useMatrix = "PeakMatrix")

counts <- peakMatrix@assays@data$PeakMatrix
df_rangers <- as.data.frame(peakMatrix@rowRanges@ranges)

rownames(counts) <- paste(peakMatrix@rowRanges@seqnames,
                          df_rangers$start,
                          df_rangers$end,
                          sep = "_") 

saveRDS(counts, file = "../data/VisiumHeart/PeakMatrix.Rds")

saveArchRProject(ArchRProj = proj, 
                 load = FALSE)
sessionInfo()






