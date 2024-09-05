suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(platetools))
suppressPackageStartupMessages(library(patchwork))

figure_dir <- "../figures/supplementary"
output_supp_figure_2 <- file.path(figure_dir, "supp_figure_2_platemaps.png")

# Path to platemaps
platemap_dir <- file.path(
    "../../../nf1_cellpainting_data/0.download_data/metadata"
)

# Load data
plate_A_B_file <- file.path(platemap_dir, "platemap_NF1_plate3.csv")

plate_A_B_df <- readr::read_csv(plate_A_B_file)

# Remove rows where genotype is "HET"
plate_A_B_df <- plate_A_B_df %>%
  filter(!grepl("HET", genotype))

dim(plate_A_B_df)
head(plate_A_B_df)

platemap_A_B <-
        platetools::raw_map(
            data = as.numeric(plate_A_B_df$seed_density),
            well = plate_A_B_df$well_position,
            plate = 96,
            size = 8
        ) +
        theme(plot.title = element_text(size = 10, face = "bold")) +
        ggplot2::geom_point(aes(shape = plate_A_B_df$genotype)) +
        ggplot2::scale_shape_discrete(name = "Genotype") +
        ggplot2::scale_fill_gradient2(
        name = "Seed Density",
        low = "white",
        high = "red",
        ) 

platemap_A_B

# Load data
plate_C_file <- file.path(platemap_dir, "platemap_NF1_plate5.csv")

plate_C_df <- readr::read_csv(plate_C_file)

# Remove rows where genotype is "HET"
plate_C_df <- plate_C_df %>%
  filter(!grepl("HET", genotype))

dim(plate_C_df)
head(plate_C_df)

platemap_C <-
        platetools::raw_map(
            data = plate_C_df$genotype,
            well = plate_C_df$well_position,
            plate = 96,
            size = 8
        ) +
        theme(plot.title = element_text(size = 10, face = "bold")) +
        ggplot2::scale_fill_discrete(name = "Genotype")  

platemap_C

align_plot <- (
    platemap_A_B /
    platemap_C 
) + plot_layout(heights= c(2,2))

align_plot

supp_fig_2_gg <- (
  align_plot
) + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 20))

# Save or display the plot
ggsave(output_supp_figure_2, plot = supp_fig_2_gg, dpi = 500, height = 7.75, width = 7)

supp_fig_2_gg
