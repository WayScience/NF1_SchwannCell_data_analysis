suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(RColorBrewer))

load_image <- function(path){
    img <- png::readPNG(path)
    # Convert the image to a raster object
    g <- grid::rasterGrob(img, interpolate=TRUE)

    # Create a ggplot
    p <- ggplot() +
    annotation_custom(g, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
    theme_void()
    return(p)
}

# Directory with single-cell crops
sc_crop_dir <- "./sc_crops"

# Path to each composite image (min or max) per top feature
max_Null_feat1 <- file.path(sc_crop_dir, "max_Null_feature_1", "max_Null_feature_1_composite_cropped.png")
max_Null_feat2 <- file.path(sc_crop_dir, "max_Null_feature_2", "max_Null_feature_2_composite_cropped.png")
max_Null_feat3 <- file.path(sc_crop_dir, "max_Null_feature_3", "max_Null_feature_3_composite_cropped.png")
min_Null_feat1 <- file.path(sc_crop_dir, "min_Null_feature_1", "min_Null_feature_1_composite_cropped.png")
min_Null_feat2 <- file.path(sc_crop_dir, "min_Null_feature_2", "min_Null_feature_2_composite_cropped.png")
min_Null_feat3 <- file.path(sc_crop_dir, "min_Null_feature_3", "min_Null_feature_3_composite_cropped.png")

max_WT_feat1 <- file.path(sc_crop_dir, "max_WT_feature_1", "max_WT_feature_1_composite_cropped.png")
max_WT_feat2 <- file.path(sc_crop_dir, "max_WT_feature_2", "max_WT_feature_2_composite_cropped.png")
max_WT_feat3 <- file.path(sc_crop_dir, "max_WT_feature_3", "max_WT_feature_3_composite_cropped.png")
min_WT_feat1 <- file.path(sc_crop_dir, "min_WT_feature_1", "min_WT_feature_1_composite_cropped.png")
min_WT_feat2 <- file.path(sc_crop_dir, "min_WT_feature_2", "min_WT_feature_2_composite_cropped.png")
min_WT_feat3 <- file.path(sc_crop_dir, "min_WT_feature_3", "min_WT_feature_3_composite_cropped.png")

# load top Null feat images 
max_Null_feat1_image <- load_image(max_Null_feat1)
max_Null_feat2_image <- load_image(max_Null_feat2)
max_Null_feat3_image <- load_image(max_Null_feat3)
min_Null_feat1_image <- load_image(min_Null_feat1)
min_Null_feat2_image <- load_image(min_Null_feat2)
min_Null_feat3_image <- load_image(min_Null_feat3)

# load top WT feat images 
max_WT_feat1_image <- load_image(max_WT_feat1)
max_WT_feat2_image <- load_image(max_WT_feat2)
max_WT_feat3_image <- load_image(max_WT_feat3)
min_WT_feat1_image <- load_image(min_WT_feat1)
min_WT_feat2_image <- load_image(min_WT_feat2)
min_WT_feat3_image <- load_image(min_WT_feat3)


# Create list of images
list_of_images <- list(
    max_Null_feat1_image,
    max_Null_feat2_image,
    max_Null_feat3_image,
    min_Null_feat1_image,
    min_Null_feat3_image,
    min_Null_feat3_image,

    max_WT_feat1_image,
    max_WT_feat2_image,
    max_WT_feat3_image,
    min_WT_feat1_image,
    min_WT_feat2_image,
    min_WT_feat3_image
)

width <- 2.5
height <- 2.5

text_size <- 8

options(repr.plot.width = width, repr.plot.height = height)

# blank
blank <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = ""), size = text_size) 
    + theme_void()
)

# ggplot of just text for labelling min versus max cells
WT_min_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Minimum values\n(WT cells)"), size = text_size) 
    + theme_void()
)
Null_min_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Minimum values\n(Null cells)"), size = text_size) 
    + theme_void()
)
WT_max_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Maximum values\n(WT cells)"), size = text_size) 
    + theme_void()
)
Null_max_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Maximum values\n(Null cells)"), size = text_size) 
    + theme_void()
)

# patchwork the cropped single-cell images together
width <- 17
height <- 8

options(repr.plot.width = width, repr.plot.height = height)

# stich the images together for each genotype
top_max_null_feat_images <- (
    Null_max_text
    + list_of_images[[1]]
    + list_of_images[[2]]
    + list_of_images[[3]]
    + plot_layout(nrow = 1)
)
top_min_null_feat_images <- (
    WT_min_text
    + list_of_images[[4]]
    + list_of_images[[5]]
    + list_of_images[[6]]
    + plot_layout(nrow = 1)
)
top_max_WT_feat_images <- (
    WT_max_text
    + list_of_images[[7]]
    + list_of_images[[8]]
    + list_of_images[[9]]
    + plot_layout(nrow = 1)
)
top_min_WT_feat_images <- (
    Null_min_text
    + list_of_images[[10]]
    + list_of_images[[11]]
    + list_of_images[[12]]
    + plot_layout(nrow = 1)
)

# Generate labels for each plot with CellProfiler feature
width <- 2.5
height <- 2.5

text_size <- 10

options(repr.plot.width = width, repr.plot.height = height)

# ggplot of just text
top_null_feat_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Cytoplasm_RadialDistribution_FracAtD_RFP_4of4"), size = text_size) 
    + theme_void()
)
top_WT_feat_text <- (
    ggplot()
    + geom_text(aes(x = 0.5, y = 0.5, label = "Cells_Correlation_Correlation_DAPI_GFP"), size = text_size) 
    + theme_void()
)

# patch feature texts together
top_null_patch_text <- (
    top_null_feat_text
    + plot_layout(nrow = 1)
)

top_WT_patch_text <- (
    top_WT_feat_text
    + plot_layout(nrow = 1)
)

width <- 17
height <- 2.5

options(repr.plot.width = width, repr.plot.height = height)


top_WT_patch_text

# Create montage
width <- 14.5
height <- 8

options(repr.plot.width = width, repr.plot.height = height)

# patch the images together
top_null_feat_plot <- (
    wrap_elements(full = top_null_patch_text)
    + wrap_elements(top_max_null_feat_images)
    + wrap_elements(top_min_null_feat_images)
    + plot_layout(ncol = 1, heights = c(0.2, 1, 1))
    )

top_null_feat_plot

# save plot
ggsave(
    file.path(
        paste0(
            "./","top_Null_feature_montage.png"
        )
    ),
    top_null_feat_plot, width = width, height = height, dpi = 600
)

# Create montage
width <- 14.5
height <- 8

options(repr.plot.width = width, repr.plot.height = height)

# patch the images together
top_wt_feat_plot <- (
    wrap_elements(full = top_WT_patch_text)
    + wrap_elements(top_max_WT_feat_images)
    + wrap_elements(top_min_WT_feat_images)
    + plot_layout(ncol = 1, heights = c(0.2, 1, 1))
    )

top_wt_feat_plot

# save plot
ggsave(
    file.path(
        paste0(
            "./","top_WT_feature_montage.png"
        )
    ),
    top_wt_feat_plot, width = width, height = height, dpi = 600
)
