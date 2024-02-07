# adapters sets for fusion

GUE_human_tfs = ["human_0_tf_100", 
                 # "human_1_tf_100", 
                 "human_2_tf_100", 
                 "human_3_tf_100", 
                 "human_4_tf_100"]
GUE_mouse_tfs = ["mouse_0_tf_100", 
                 "mouse_1_tf_100", 
                 "mouse_2_tf_100", 
                 "mouse_3_tf_100",
                 "mouse_4_tf_100"]
GUE_all_tfs = GUE_human_tfs + GUE_mouse_tfs
GUE_human_promoters = ["human_all-core_promoters_300", 
                       "human_all_promoters_300", 
                       "human_nontata-core_promoters_300", 
                       "human_nontata_promoters_300", 
                       "human_tata-core_promoters_300", 
                       "human_tata_promoters_300",]
GUE_yeast_EMP = ["yeast_H3_EMP_500", 
                 "yeast_H3K14ac_EMP_500", 
                 "yeast_H3K36me3_EMP_500", 
                 "yeast_H3K4me1_EMP_500", 
                 "yeast_H3K4me3_EMP_500", 
                 "yeast_H3K79me3_EMP_500", 
                 "yeast_H3K9ac_EMP_500", 
                 "yeast_H4_EMP_500", 
                 "yeast_H4ac_EMP_500", 
                 "yeast_H3K4me2_EMP_500",]
GUE_human_splicing = ["human_reconstructed_splicing_400"]
GUE_tfs_promoters = GUE_all_tfs + GUE_human_promoters
GUE_tfs_promoters_splicing = GUE_all_tfs + GUE_human_promoters + GUE_human_splicing
GUE_all = GUE_tfs_promoters_splicing + GUE_yeast_EMP
GUE_tfs_promoters_splicing = GUE_all_tfs + GUE_human_promoters + GUE_human_splicing
GUE_all = GUE_tfs_promoters_splicing + GUE_yeast_EMP



# minimal tasks sets

min_tfs_set = ["mouse_4_tf_100", "mouse_0_tf_100", "human_3_tf_100", "human_2_tf_100"] # "mouse_3_tf_100",
min_mouse_tfs_set = ["mouse_3_tf_100", "mouse_4_tf_100"]
min_human_tfs_set = ["human_3_tf_100", "human_2_tf_100"]
min_promoters_set = ["human_tata_promoters_300_adapter", "human_all-core_promoters_300",
                    "human_nontata-core_promoters_300", "human_tata-core_promoters_300" ]
min_EMP_set = ["yeast_H3K4me2_EMP_500",
               "yeast_H3K4me3_EMP_500",
               "yeast_H3K14ac_EMP_500",
               "yeast_H4ac_EMP_500",
               "yeast_H3K9ac_EMP_500",
               "yeast_H3K36me3_EMP_500",
               "yeast_H3K4me1_EMP_500",
               # "yeast_H3_EMP_500",  
               # "yeast_H3K79me3_EMP_500", 
               # "yeast_H4_EMP_500", 
               ]

splicing = ["human_reconstructed_splicing_400"]