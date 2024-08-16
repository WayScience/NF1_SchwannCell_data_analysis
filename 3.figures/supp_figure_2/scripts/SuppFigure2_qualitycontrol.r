suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(arrow))

figure_dir <- "../figures/supplementary"
output_supp_figure_2 <- file.path(figure_dir, "supp_figure_2_qualitycontrol.png")

# Path to UMAP results
qc_results_dir <- file.path(
    "../../../nf1_cellpainting_data/1.cellprofiler_ic/image_quality_control"
)

# Load data
qc_results_file <- file.path(qc_results_dir, "concat_img_quality_data.parquet")

qc_results_df <- arrow::read_parquet(qc_results_file)

# Remove any plate 4 image sets
qc_results_df <- qc_results_df %>% dplyr::filter(Metadata_Plate != "Plate_4")

dim(qc_results_df)
head(qc_results_df)

width <- 18
height <- 6
options(repr.plot.width = width, repr.plot.height = height)

# Ensure the Channel factor has the correct order
qc_results_df$Channel <- factor(qc_results_df$Channel, levels = c("DAPI", "GFP", "CY5", "RFP"))

# Threshold values
upper_threshold <- -1.36839084751248
lower_threshold <- -2.4936740062477365

# Create the plot with the specified color mapping
qc_blur_plot <- ggplot(qc_results_df, aes(x = ImageQuality_PowerLogLogSlope, fill = Channel)) +
    geom_density(alpha = 0.5, adjust = 1.5) + 
    scale_fill_manual(
        values = c("DAPI" = "#4472C4",  # Blue
                   "GFP" = "#70AD47",  # Green
                   "CY5" = "#D35B9D",  # Brighter magenta
                   "RFP" = "#b42718")  # Red
    ) +
    geom_vline(aes(xintercept = upper_threshold, color = paste("Upper =", round(upper_threshold, 2))), 
               linetype = "dashed", linewidth = 1.2) + 
    geom_vline(aes(xintercept = lower_threshold, color = paste("Lower =", round(lower_threshold, 2))), 
               linetype = "dashed", linewidth = 1.2) +  
    scale_color_manual(name = "QC Thresholds", values = c("Upper = -1.37" = "red", 
                                                          "Lower = -2.49" = "red")) +  # Set the color and label for the legend
    labs(
        x = "PowerLogLogSlope",
        y = "Density"
    ) +
    theme_bw() +
    facet_wrap(~ Metadata_Plate) +  # Facet by plate
    theme(
        strip.text = element_text(size = 16),   # Facet titles
        axis.title = element_text(size = 16),  # Axis titles
        axis.text = element_text(size = 14),   # Axis text
        legend.title = element_text(size = 16), # Legend title
        legend.text = element_text(size = 14)   # Legend text
    )

qc_blur_plot


width <- 15
height <- 8
options(repr.plot.width = width, repr.plot.height = height)

# Take natural logarithm plus one of the percent maximal for better visualization
qc_results_df <- qc_results_df %>%
  mutate(LogImageQuality = log1p(ImageQuality_PercentMaximal))

# Create a label for the threshold line
original_threshold <- 1.4

# Create the plot
qc_saturation_plot <- ggplot(qc_results_df, aes(x = LogImageQuality)) +
  geom_histogram(binwidth = 0.1, fill = "lightblue", color = "black") +
  facet_wrap(~ Metadata_Plate) +
  geom_vline(aes(xintercept = log1p(original_threshold), 
                 color = paste("Threshold =", original_threshold, "%")),
             linetype = "dashed", size = 1) +
  scale_color_manual(name = "QC Threshold", values = "red") + # Customize the legend title and color
  labs(
    x = "Natural Logarithm of (Image Quality Percentage + 1)",
    y = "Count"
  ) +
  theme_bw() +
  theme(
    strip.text = element_text(size = 16),   # Facet titles
    axis.title = element_text(size = 16),  # Axis titles
    axis.text = element_text(size = 14),   # Axis text
    legend.title = element_text(size = 16), # Legend title
    legend.text = element_text(size = 14)   # Legend text
  )

qc_saturation_plot

align_plot <- (
    qc_blur_plot /
    qc_saturation_plot 
) + plot_layout(heights= c(2,2))

align_plot

supp_fig_2_gg <- (
  align_plot
) + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 25))

# Save or display the plot
ggsave(output_supp_figure_2, plot = supp_fig_2_gg, dpi = 500, height = 8, width = 12)

supp_fig_2_gg
