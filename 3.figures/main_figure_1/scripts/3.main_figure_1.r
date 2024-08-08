suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(RColorBrewer))

figure_dir <- "../figures"
output_main_figure_1 <- file.path(figure_dir, "main_figure_1_workflow.png")

workflow_path = file.path("../figures/workflow.png")
workflow_img = png::readPNG(workflow_path)

# Get the dimensions of the image
img_height <- nrow(workflow_img)
img_width <- ncol(workflow_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the workflow image from BioRender to a ggplot object
workflow <- ggplot() +
  annotation_custom(
    rasterGrob(workflow_img, interpolate = TRUE),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  theme_void() +
  coord_fixed(ratio = aspect_ratio, clip = "off") +
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))  # Adjust margins as needed

workflow

montage_path = file.path("../figures/all_genotypes_montage.png")
montage_img = png::readPNG(montage_path)

# Get the dimensions of the image
img_height <- nrow(montage_img)
img_width <- ncol(montage_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the workflow image from BioRender to a ggplot object
genotype_montage <- ggplot() +
  annotation_custom(
    rasterGrob(montage_img, interpolate = TRUE),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  theme_void() +
  coord_fixed(ratio = aspect_ratio, clip = "off") +
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))  # Adjust margins as needed

genotype_montage

fig_1_gg <- (
  genotype_montage /
  workflow
) + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 30))

# Save or display the plot
ggsave(output_main_figure_1, plot = fig_1_gg, dpi = 500, height = 14, width = 14)

fig_1_gg
