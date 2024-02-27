suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(platetools))

figure_dir <- "figures/supplementary"
output_supp_figure_2 <- file.path(figure_dir, "supp_figure_2_plate4_platemaps.png")

url <- "https://raw.githubusercontent.com/WayScience/nf1_cellpainting_data/main/0.download_data/metadata/platemap_NF1_plate4.csv"
plate_4_df <- read.csv(url)

dim(plate_4_df)
head(plate_4_df)

platemap_dose <- platetools::raw_map(
    data = as.character(plate_4_df$Concentration),
    well = plate_4_df$well_position,
    plate = 96,
    size = 8
) +
ggtitle(paste("siRNA treatment and dose platemap")) +
theme(plot.title = element_text(size = 12, face = "bold")) +
ggplot2::geom_point(aes(shape = plate_4_df$siRNA)) +
ggplot2::scale_shape_discrete(
    name = "siRNA Treatments",
    limits = c("None", "Scramble", "NF1 Target 1", "NF1 Target 2"),
    guide = guide_legend(override.aes = list(size = 3))  # Adjust size here
) +
ggplot2::scale_fill_manual(
    name = "Concentrations (nM)",
    values = c("#ffffff", "#d9f0d9", "#a3e8a3", "#6ed46e", "#3ab93a", "#007a00")
) +
theme(
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.position = "right",
    # move legend around so it fits better on the screen
    legend.margin = margin(-15, 0, 10, 0),
    legend.box = "vertical",
    axis.text.x = element_text(size = 10),  # Adjust x-axis tick size
    axis.text.y = element_text(size = 10)   # Adjust y-axis tick size
)

platemap_dose


# Platemap for genotype
platemap_genotype <- platetools::raw_map(
    data = plate_4_df$genotype,
    well = plate_4_df$well_position,
    plate = 96,
    size = 8
) +
ggtitle(paste("Genotype platemap")) +
theme(plot.title = element_text(size = 12, face = "bold")) +
ggplot2::scale_fill_discrete(name = "Genotype") +
theme(
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.position = "right",
    # move legend around so it fits better on the screen
    legend.margin = margin(-10, 0, 10, 0),
    legend.box = "horizontal",
    axis.text.x = element_text(size = 10),  # Adjust x-axis tick size
    axis.text.y = element_text(size = 10)   # Adjust y-axis tick size
)

platemap_genotype


align_plot_gg <- (
    platemap_dose /
    platemap_genotype
) + plot_layout(heights = c(1, 1))

supp_fig_2_gg <- (
  align_plot_gg
) + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 15))

# Save or display the plot
ggsave(output_supp_figure_2, dpi = 500, height = 8, width = 8)

supp_fig_2_gg
