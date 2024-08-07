suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(grid))
suppressPackageStartupMessages(library(patchwork))
suppressPackageStartupMessages(library(RColorBrewer))

figure_dir <- "../figures"
output_main_figure_4 <- file.path(
    figure_dir, "main_figure_4_feature_importance.png"
)
results_dir <- file.path(
    "../../2.evaluate_model/model_evaluation_data"
)

# Load data
feat_import_file <- file.path(results_dir, "feature_importances.parquet")

feat_import_df <- arrow::read_parquet(feat_import_file)

dim(feat_import_df)
head(feat_import_df)

# Split out components of feature name for visualization
feat_import_df <- feat_import_df %>%
    dplyr::arrange(desc(abs(feature_importances))) %>%
    tidyr::separate(
        feature_names,
        into = c(
            "compartment",
            "feature_group",
            "measurement",
            "channel", 
            "parameter1", 
            "parameter2",
            "parameter3"
        ),
        sep = "_",
        remove = FALSE
    ) %>%
    dplyr::mutate(channel_cleaned = channel) %>%
    dplyr::mutate(parameter1_cleaned = parameter1)

# Convert the feature_importances to the absolute value
feat_import_df <- feat_import_df %>%
  mutate(feature_importances = abs(feature_importances))

feat_import_df$channel_cleaned <- dplyr::recode(feat_import_df$channel_cleaned,
    "DAPI" = "Nucleus",
    "GFP" = "ER",
    "RFP" = "Actin",
    "CY5" = "Mito",
    .default = "other",
    .missing = "other"
)

feat_import_df$parameter1_cleaned <- dplyr::recode(feat_import_df$parameter1_cleaned,
    "DAPI" = "Nucleus",
    "GFP" = "ER",
    "RFP" = "Actin",
    "CY5" = "Mito",
    .default = "other",
    .missing = "other"
)


print(dim(feat_import_df))
head(feat_import_df, 3)

channels <- c(
    "Mito" = "Mito",
    "Nucleus" = "DNA",
    "ER" = "ER",
    "Actin" = "Actin",
    "other" = "other"
)

# Find top feature
top_feat_import_df <- feat_import_df %>%
    dplyr::filter(channel_cleaned %in% names(channels)) %>%
    dplyr::group_by(feature_group, channel_cleaned, compartment) %>%
    dplyr::slice_max(order_by = feature_importances, n = 1)

# Add rounded coefficient values to the data frame
top_feat_import_df <- top_feat_import_df %>%
    mutate(rounded_coeff = round(feature_importances, 2))

# Reorder the channel_cleaned factor levels
channel_order <- c("Nucleus", "ER", "Mito", "Actin", "other")
top_feat_import_df <- top_feat_import_df %>%
    mutate(channel_cleaned = factor(channel_cleaned, levels = channel_order))

# Process data for plotting
other_feature_group_df <- top_feat_import_df %>%
    dplyr::filter(!feature_group %in% c("AreaShape", "Correlation", "Neighbors", "Location"))

# Create a new data frame for the red star in the Cytoplasm facet for Actin and RadialDistribution
red_box_radial <- other_feature_group_df %>%
    dplyr::filter(channel_cleaned == "Actin" & feature_group == "RadialDistribution" & compartment == "Cytoplasm")

width <- 12
height <- 6
options(repr.plot.width = width, repr.plot.height = height)

# Create the plot with stars for Actin and RadialDistribution, and ER and Intensity in the Cytoplasm facet
feature_importance_gg <- (
    ggplot(other_feature_group_df, aes(x = channel_cleaned, y = feature_group))
    + geom_point(aes(fill = feature_importances), pch = 22, size = 27)
    + geom_text(aes(label = rounded_coeff), size = 6)
    + geom_point(data = red_box_radial, 
                aes(x = channel_cleaned, y = feature_group), 
                color = "red", 
                shape = 0, 
                size = 25, 
                stroke = 1.5) # Red box for Actin and RadialDistribution
    + facet_wrap("~compartment", ncol = 3)
    + theme_bw()
    + scale_fill_distiller(
        name = "Top absolute value\nweight from model",
        palette = "YlGn",
        direction = 1,
        limits = c(0, 2.5)
    )
    + xlab("Channel")
    + ylab("Feature group")
    + theme(
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, size = 16, vjust = 0.6, hjust = 0.5),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 18),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
        ),
        legend.position = "bottom", # Move legend to bottom
        legend.title = element_text(size = 16, margin = margin(b = 15)), # Increase space between title and gradient
        legend.text = element_text(size = 14),
        legend.key.height = unit(1.9, "cm"), # Increase height of legend key
        legend.key.width = unit(2, "cm"), # Optionally, increase width of legend key
        legend.margin = margin(t = 35),
    )
)

feature_importance_gg

# Filter data to include AreaShape and Neighbors feature groups
area_shape_neighbors_df <- feat_import_df %>%
    dplyr::filter(feature_group %in% c("AreaShape", "Neighbors", "Location")) %>%
    dplyr::mutate(area_shape_indicator = paste(measurement, channel, parameter1, sep = "_"),
                  measurement = ifelse(measurement == "SecondClosestDistance", "SCD", measurement))

# Add rounded coefficient values to the data frame
area_shape_neighbors_df <- area_shape_neighbors_df %>%
    mutate(rounded_coeff = round(feature_importances, 2))

# Find top feature per measurement
top_area_shape_neighbors_df <- area_shape_neighbors_df %>%
    dplyr::filter(channel_cleaned %in% names(channels)) %>%
    dplyr::group_by(feature_group, channel_cleaned, compartment) %>%
    dplyr::slice_max(order_by = feature_importances, n = 1)

width <- 12
height <- 6
options(repr.plot.width = width, repr.plot.height = height)

# Create the plot
areashape_neighbors_importance_gg <- (
    ggplot(top_area_shape_neighbors_df, aes(x = feature_group, y = measurement))
    + geom_point(aes(fill = feature_importances), pch = 22, size = 18)
    + geom_text(aes(label = rounded_coeff), size = 6)
    + facet_wrap("~compartment", ncol = 3)
    + theme_bw()
    + scale_fill_distiller(
        name = "Top absolute\nvalue weight\nfrom model",
        palette = "YlGn",
        direction = 1,
        limits = c(0, 2.5)
    )
    + xlab("Feature group")
    + ylab("Measurement")
    + theme(
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, size = 16, vjust = 0.6, hjust = 0.5),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 18),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
        ),
        legend.position = "none"
    )
)

areashape_neighbors_importance_gg

correlation_df <- feat_import_df %>% dplyr::filter(feature_group  == "Correlation")

# Add rounded coefficient values to the data frame
correlation_df <- correlation_df %>%
    mutate(rounded_coeff = round(feature_importances, 2))

# Find top feature per channel combo
top_correlation_df <- correlation_df %>%
    dplyr::filter(channel_cleaned %in% names(channels)) %>%
    dplyr::group_by(feature_group, channel_cleaned, compartment, channel, parameter1) %>%
    dplyr::slice_max(order_by = feature_importances, n = 1)

# Create a new data frame for the red star in the Cytoplasm facet for Actin and RadialDistribution
red_box_dapi_er_corr <- top_correlation_df %>%
    dplyr::filter(channel_cleaned == "Nucleus" & parameter1_cleaned == "ER" & compartment == "Cells")

width <- 12
height <- 6
options(repr.plot.width = width, repr.plot.height = height)

# Create the plot
correlation_importance_gg <- (
    ggplot(top_correlation_df, aes(x = channel_cleaned, y = parameter1_cleaned))
    + geom_point(aes(fill = feature_importances), pch = 22, size = 18)
    + geom_text(aes(label = rounded_coeff), size = 6)
    + geom_point(data = red_box_dapi_er_corr, 
                aes(x = channel_cleaned, y = parameter1_cleaned), 
                color = "red", 
                shape = 0, 
                size = 16, 
                stroke = 1.5) # Red box for Correlation DAPI and ER
    + facet_wrap("~compartment", ncol = 3)
    + theme_bw()
    + scale_fill_distiller(
        name = "Top absolute\nvalue weight\nfrom model",
        palette = "YlGn",
        direction = 1,
        limits = c(0, 2.5)
    )
    + xlab("Channel (correlation)")
    + ylab("Channel (correlation)")
    + theme(
        axis.text = element_text(size = 16),
        axis.text.x = element_text(angle = 45, size = 16, vjust = 0.6, hjust = 0.5),
        axis.title = element_text(size = 18),
        strip.text = element_text(size = 18),
        strip.background = element_rect(
            colour = "black",
            fill = "#fdfff4"
        ),
        legend.position = "none"
    )
)

correlation_importance_gg

left_plot <- (
    areashape_neighbors_importance_gg /
    correlation_importance_gg
) + plot_layout(heights = c(1,1.4))

left_plot

coefficient_plot <- (
    left_plot |
    plot_spacer() |
    free(feature_importance_gg)
) + plot_layout(widths = c(1,0.05,1.5))

ggsave("./coefficient_plot.png", coefficient_plot, width=21.5, height=7, dpi=500)

coefficient_plot

corr_feat_path = file.path("./correlation_feature_montage.png")
corr_feat_img = png::readPNG(corr_feat_path)

# Get the dimensions of the image
img_height <- nrow(corr_feat_img)
img_width <- ncol(corr_feat_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the image montage to a ggplot object
corr_montage <- ggplot() +
  annotation_custom(
    rasterGrob(corr_feat_img, interpolate = TRUE),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  theme_void() +
  coord_fixed(ratio = aspect_ratio, clip = "off") +
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))  # Adjust margins as needed

corr_montage

radial_path = file.path("./radial_feature_montage.png")
radial_img = png::readPNG(radial_path)

# Get the dimensions of the image
img_height <- nrow(radial_img)
img_width <- ncol(radial_img)

# Calculate the aspect ratio
aspect_ratio <- img_height / img_width

# Plot the image montage to a ggplot object
radial_montage <- ggplot() +
  annotation_custom(
    rasterGrob(radial_img, interpolate = TRUE),
    xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf
  ) +
  theme_void() +
  coord_fixed(ratio = aspect_ratio, clip = "off") +
  theme(plot.margin = margin(0, 0, 0, 0, "cm"))  # Adjust margins as needed

radial_montage

bottom_montage <- (
   free(corr_montage) | 
   radial_montage
) + plot_layout(widths = c(1,1.25))

bottom_montage

align_plot <- (
    coefficient_plot /
    bottom_montage
) + plot_layout(heights = c(1,1))

align_plot

fig_4_gg <- (
  align_plot
) + plot_annotation(tag_levels = list(c("A", "", "", "B", "C"))) & theme(plot.tag = element_text(size = 25))

# Save or display the plot
ggsave(output_main_figure_4, plot = fig_4_gg, dpi = 500, height = 15, width = 23.5)

fig_4_gg
