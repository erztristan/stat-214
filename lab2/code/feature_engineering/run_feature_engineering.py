import feature_engineering

feature_engineering.feature_engineering(ae_train = False, 
                                        path_labeled = "../../data/labeled_data",
                                        path_unlabeled = "../../data/unlabeled_data", 
                                        path_config = "configs/default.yaml",
                                        path_output = "../../data/data_with_features_and_pca",
                                    )

feature_engineering.feature_engineering(ae_train = False,
                                        path_labeled = "../../data/unlabeled_data_prediction",
                                        path_unlabeled = "../../data/unlabeled_data",
                                        path_config = "configs/default.yaml",
                                        path_output = "../../data/data_with_features_and_pca_unlabeled"
                                    )