suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(RColorBrewer))

figure_dir <- "../figures"
output_main_figure_2 <- file.path(
    figure_dir, "main_figure_2_UMAP_sc_correlations.png"
)

# Path to UMAP results
UMAP_results_dir <- file.path(
    "../../../nf1_cellpainting_data/4.analyze_data/notebooks/UMAP/results/"
)

# Load data
UMAP_results_file <- file.path(UMAP_results_dir, "UMAP_concat_model_plates_sc_feature_selected.tsv")

UMAP_results_df <- readr::read_tsv(UMAP_results_file)

dim(UMAP_results_df)
head(UMAP_results_df)

width <- 8
height <- 10
options(repr.plot.width = width, repr.plot.height = height)

umap_fig_gg <- (
  ggplot(UMAP_results_df, aes(x = UMAP0, y = UMAP1))
  + geom_point(
      aes(color = Metadata_genotype),
      size = 0.2,
      alpha = 0.4
  )
  + theme_bw()
  + guides(
      color = guide_legend(
          override.aes = list(size = 2)
      )
  )
  + labs(x = "UMAP0", y = "UMAP1", color = "NF1\ngenotype")
  # change the text size
  + theme(
      strip.text = element_text(size = 18),
      # x and y axis text size
      axis.text.x = element_text(size = 18),
      axis.text.y = element_text(size = 18),
      # x and y axis title size
      axis.title.x = element_text(size = 18),
      axis.title.y = element_text(size = 18),
      # legend text size
      legend.text = element_text(size = 18),
      legend.title = element_text(size = 18)
  )
)

umap_fig_gg

# Replace missing values in Metadata_genotype with "Null"
UMAP_results_df$Metadata_genotype <- na_if(UMAP_results_df$Metadata_genotype, "")

# Group by Metadata_genotype and summarize the count of rows per group across all plates
total_counts_per_genotype <- UMAP_results_df %>%
    group_by(Metadata_genotype) %>%
    summarize(count = n(), .groups = 'drop') %>%
    mutate(Metadata_Plate = "All_plates")

# Group by Metadata_genotype and Metadata_Plate, then summarize the count of rows per group
counts_df <- UMAP_results_df %>%
    group_by(Metadata_genotype, Metadata_Plate) %>%
    summarize(count = n(), .groups = 'drop')

# Combine the counts per genotype across all plates with the counts per genotype and plate
combined_counts_df <- bind_rows(counts_df, total_counts_per_genotype)

# Confirm any NA values are "Null" strings in Metadata_genotype column
combined_counts_df$Metadata_genotype[is.na(combined_counts_df$Metadata_genotype)] <- "Null"

# View the resulting counts dataframe
dim(combined_counts_df)
combined_counts_df


# Create the histogram plot with adjusted dodge width
histogram_plot <- ggplot(combined_counts_df, aes(x = Metadata_Plate, y = count, fill = Metadata_genotype)) +
    geom_bar(stat = "identity", position = position_dodge(width = 1.0)) +  # Adjust dodge width
    geom_text(aes(label = count), position = position_dodge(width = 1.0), vjust = -0.5, size = 5) +  # Adjust dodge width
    labs(x = "Plate", y = "Count", fill = "NF1\ngenotype") +
    ylim(0, 15000) +  # Adjust y-axis limit if needed
    theme_bw() +
    theme(
        # x and y axis text size
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18),
        # x and y axis title size
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        # legend text size
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 18),
    )

histogram_plot

# Path to correlation per plate results
corr_results_dir <- file.path(
    "../../0.data_analysis/construct_phenotypic_expression_plate_4_fs_data/median_correlation_relationships/post_fs_aggregation_correlations/construct_correlation_data"
)

# Load data
corr_results_file <- file.path(corr_results_dir, "concatenated_all_plates_correlations.parquet")

corr_results_df <- arrow::read_parquet(corr_results_file)

# Add a new column `same_genotype` to check if the correlation row is comparing between the same genotype
corr_results_df$same_genotype <- corr_results_df$Metadata_genotype__group0 == corr_results_df$Metadata_genotype__group1

# Add a new column `same_plate` to check if the correlation row is comparing between the same plate
corr_results_df$same_plate <- corr_results_df$Metadata_plate__group0 == corr_results_df$Metadata_plate__group1

dim(corr_results_df)
head(corr_results_df)

focus_corr_colors = c(
    "TRUE" = "blue",
    "FALSE" = "orange"
)
focus_corr_labels  = c(
    "TRUE" = "Yes",
    "FALSE" = "No"
)
facet_labels  = c(
    "TRUE" = "Same plate",
    "FALSE" = "Different plate"
)

width <- 8
height <- 10
options(repr.plot.width = width, repr.plot.height = height)

genotype_corr_gg <- (
    ggplot(corr_results_df, aes(x = correlation))
    + geom_density(aes(fill = same_genotype), alpha = 0.5)
    + facet_grid("~same_plate", labeller = as_labeller(facet_labels))
    + scale_fill_manual(
        "Is the\npairwise\ncomparison\nfrom the\nsame genotype?",
        values = focus_corr_colors,
        label = focus_corr_labels
    )
    + guides(
        color = guide_legend(
            override.aes = list(size = 2)
        )
    )
    + labs(x = "pairwise Pearson correlation", y = "Density")
    + geom_vline(xintercept = 0, linetype = "dashed", color = "darkgrey")
    + xlim(-1, 1.05)
    + theme_bw()
    # change the text size
    + theme(
        strip.text = element_text(size = 16),
        # x and y axis text size
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18),
        # x and y axis title size
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        # legend text size
        legend.text = element_text(size = 18),
        legend.title = element_text(size = 18),
    )
)

genotype_corr_gg

bottom_plot <- (
    free(umap_fig_gg) |
    genotype_corr_gg
) + plot_layout(widths = c(2.5,2))

bottom_plot

align_plot <- (
    free(histogram_plot) /
    bottom_plot
) + plot_layout(heights = c(2.25,2))

align_plot

fig_2_gg <- (
  align_plot
) + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(size = 25))

# Save or display the plot
ggsave(output_main_figure_2, plot = fig_2_gg, dpi = 500, height = 10, width = 14)

fig_2_gg
