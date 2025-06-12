# Reproducing Evaluations

These scripts produce tables and figures in the paper:
 * `evaluate_human_tasks.py` shows results for the human evaluation protocol. Switch `USE_FINAL_DATA` to `False` to create the pilot data result. (Note that the pilot data results values are slightly different to those reported in the paper due to small changes in annotator filtering logic; all results are directionally the same)
   * Table 2: Krippendorff's α over topics
   * Figures 2a and 2b: Boxplots of inter-annotator and topic model-annotator correlations for the `Wiki` data
   * Table 5 (appendix): Krippendorff's α for the pilot data
   * Figures 5 and 6 (appendix): Boxplots with additional metrics in the appendix for both `Wiki` and `Bills`
   * Figures 7a and 7b (appendix): Boxplots of leave-one-out inter-annotator correlations on a per-topic level
   * Figure 8 (appendix): Boxplots for the pilot data
 * `evaluate_llm_agreement.py` evaluates the relationship between the LLM proxy, ProxAnn, and human results
   * Figure 3: Barplot of LLM-human correlations
 * `alt_test_tm_eval.ipynb` runs the Alt-Test of LLM substitutatibility
   * Table 3: Results of the Alt-test (where pseudo-annotators are composed at the dataset level)
   * Table 6 (appendix): As above, but where pseudo-annotators are combined at the model level
 * `elbow_method_revised.ipynb` is a demonstration of the method for sampling representative documents for each topic based on their θ values (the document-topic probabilities)
   * Figures 4a and 4b (appendix): Distribution of top theta values and the inferred thresholds above which we sample representative documents