observeEvent(input$run_model, {
  req(clean_data())
  
  data <- clean_data()
  
  index <- createDataPartition(data$Depression_Severity, p = 0.8, list = FALSE)
  train_data <- data[index, ]
  test_data <- data[-index, ]
  
  if (input$model == "Random Forest") {
    rf_model <- randomForest(
      Depression_Severity ~ Household_income + AGE + SEX_numeric + Residential_areas_numeric + 
        Underlying_disease + MARRIED_numeric + CHILD_numeric + 
        Employment_status_numeric + Economically_impacted_numeric + 
        Income_SEX + Income_AGE + Income_Employment, 
      data = train_data, 
      ntree = input$trees,
      importance = TRUE
    )
    
    rf_pred_probs <- predict(rf_model, test_data, type = "prob")
    levels_rf <- levels(test_data$Depression_Severity)
    
    output$roc_plot <- renderPlot({
      for (class in levels_rf) {
        binary_response <- ifelse(test_data$Depression_Severity == class, 1, 0)
        roc_curve <- roc(binary_response, rf_pred_probs[, class])
        plot.roc(roc_curve, main = paste("ROC Curve for Random Forest:", class), col = "blue", print.auc = TRUE, add = (class != levels_rf[1]))
      }
    })
    
    output$model_output <- renderText({
      mse <- mean((as.numeric(rf_pred_probs) - as.numeric(as.factor(test_data$Depression_Severity)))^2)
      rmse <- sqrt(mse)
      paste("Random Forest: MSE =", round(mse, 3), ", RMSE =", round(rmse, 3))
    })
    
    output$varimp_plot <- renderPlot({
      varImpPlot(rf_model, main = "Variable Importance - Random Forest")
    })
    
  } else if (input$model == "Gradient Boosting") {
    train_matrix <- train_data %>%
      dplyr::select(Household_income, AGE, SEX_numeric, Residential_areas_numeric, 
                    Underlying_disease, MARRIED_numeric, CHILD_numeric, 
                    Employment_status_numeric, Economically_impacted_numeric, 
                    Income_SEX, Income_AGE, Income_Employment) %>%
      as.matrix()
    
    test_matrix <- test_data %>%
      dplyr::select(Household_income, AGE, SEX_numeric, Residential_areas_numeric, 
                    Underlying_disease, MARRIED_numeric, CHILD_numeric, 
                    Employment_status_numeric, Economically_impacted_numeric, 
                    Income_SEX, Income_AGE, Income_Employment) %>%
      as.matrix()
    
    gbm_model <- xgboost(
      data = train_matrix, 
      label = as.numeric(as.factor(train_data$Depression_Severity)) - 1,
      nrounds = input$nrounds,
      max_depth = input$max_depth,
      eta = input$eta,
      objective = "multi:softprob",
      num_class = length(unique(train_data$Depression_Severity)),
      verbose = 0
    )
    
    xgb_pred_probs <- predict(gbm_model, test_matrix)
    xgb_pred_probs_matrix <- matrix(xgb_pred_probs, ncol = length(unique(train_data$Depression_Severity)), byrow = TRUE)
    levels_xgb <- levels(test_data$Depression_Severity)
    
    output$roc_plot <- renderPlot({
      for (class in levels_xgb) {
        binary_response <- ifelse(test_data$Depression_Severity == class, 1, 0)
        roc_curve <- roc(binary_response, xgb_pred_probs_matrix[, which(levels_xgb == class)])
        plot.roc(roc_curve, main = paste("ROC Curve for Gradient Boosting:", class), col = "red", print.auc = TRUE, add = (class != levels_xgb[1]))
      }
    })
    
    output$model_output <- renderText({
      mse <- mean((as.numeric(xgb_pred_probs_matrix) - as.numeric(as.factor(test_data$Depression_Severity)))^2)
      rmse <- sqrt(mse)
      paste("Gradient Boosting: MSE =", round(mse, 3), ", RMSE =", round(rmse, 3))
    })
    
    output$varimp_plot <- renderPlot({
      importance <- xgb.importance(feature_names = colnames(train_matrix), model = gbm_model)
      xgb.plot.importance(importance, main = "Variable Importance - Gradient Boosting")
    })
  }
})

