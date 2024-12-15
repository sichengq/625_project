library(shiny)
library(randomForest)
library(xgboost)
library(e1071)
library(caret)
library(pROC)
library(dplyr)
library(readxl)

ui <- fluidPage(
  titlePanel("Model Comparison"),
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload a Dataset (Excel)", accept = c(".xls", ".xlsx")),
      selectInput("model", "Choose a Model:", 
                  choices = c("Random Forest", "Gradient Boosting", "Support Vector Machine")),
      sliderInput("trees", "Number of Trees (Random Forest):", 
                  min = 100, max = 1000, value = 500, step = 50),
      actionButton("run_model", "Run Model")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Model Output", 
                 verbatimTextOutput("model_output"),
                 plotOutput("roc_plot")),
        tabPanel("Variable Importance", 
                 plotOutput("varimp_plot"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  dataset <- reactive({
    req(input$datafile)
    infile <- input$datafile
    tryCatch(
      {
        
        data <- read_excel(infile$datapath)
        
        colnames(data) <- c(
          "ID", "Surveys", "AGE", "SEX", "Residential_areas", "Underlying_disease", "MARRIED",
          "CHILD", "Household_income", "Employment_status", "Economically_impacted", "PHQ1st", "PHQ2nd", "PHQ3rd",
          "StateAnger1", "AngerControl1", "StateAnger2", "AngerControl2", "StateAnger3", "AngerControl3",
          "Selfdistraction1", "Activecoping1", "Denial1", "Substanceuse1", "EmotionalSupport1", 
          "InstrumentalSupport1", "BehavioralDisengagement1", "Venting1", "PositiveReframing1", 
          "Planning1", "Humor1", "Acceptance1", "Religion1", "SelfBlame1", "Selfdistraction2", 
          "Activecoping2", "Denial2", "Substanceuse2", "EmotionalSupport2", "InstrumentalSupport2", 
          "BehavioralDisengagement2", "Venting2", "PositiveReframing2", "Planning2", "Humor2", 
          "Acceptance2", "Religion2", "SelfBlame2", "Selfdistraction3", "Activecoping3", "Denial3", 
          "Substanceuse3", "EmotionalSupport3", "InstrumentalSupport3", "BehavioralDisengagement3", 
          "Venting3", "PositiveReframing3", "Planning3", "Humor3", "Acceptance3", "Religion3", "SelfBlame3"
        )
        data
      },
      error = function(e) {
        showNotification("Failed to read or process the file. Ensure the format is correct.", type = "error")
        return(NULL)
      }
    )
  })
  
  clean_data <- reactive({
    req(dataset())
    data <- dataset() %>%
      mutate(
        Household_income = ifelse(is.na(Household_income), median(Household_income, na.rm = TRUE), Household_income),
        SEX_numeric = as.numeric(as.character(SEX)),
        Residential_areas_numeric = as.numeric(as.character(Residential_areas)),
        MARRIED_numeric = as.numeric(as.character(MARRIED)),
        CHILD_numeric = as.numeric(as.character(CHILD)),
        Employment_status_numeric = as.numeric(as.character(Employment_status)),
        Economically_impacted_numeric = as.numeric(as.character(Economically_impacted)),
        Income_SEX = Household_income * SEX_numeric,
        Income_AGE = Household_income * AGE,
        Income_Employment = Household_income * Employment_status_numeric
      ) %>%
      mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
    data
  })
  
  observeEvent(input$run_model, {
    req(clean_data())
    
    data <- clean_data()
    
    index <- createDataPartition(data$PHQ1st, p = 0.8, list = FALSE)
    train_data <- data[index, ]
    test_data <- data[-index, ]
    
    if (input$model == "Random Forest") {
      rf_model <- randomForest(
        PHQ1st ~ Household_income + AGE + SEX_numeric + Residential_areas_numeric + 
          Underlying_disease + MARRIED_numeric + CHILD_numeric + 
          Employment_status_numeric + Economically_impacted_numeric + 
          Income_SEX + Income_AGE + Income_Employment, 
        data = train_data, 
        ntree = input$trees,
        importance = TRUE
      )
      
      rf_predictions <- predict(rf_model, test_data)
      mse <- mean((test_data$PHQ1st - rf_predictions)^2)
      rmse <- sqrt(mse)
      
      output$model_output <- renderText({
        paste("Random Forest: MSE =", round(mse, 3), ", RMSE =", round(rmse, 3))
      })
      
      output$varimp_plot <- renderPlot({
        varImpPlot(rf_model, main = "Variable Importance - Random Forest")
      })
      
      output$roc_plot <- renderPlot({
        plot(test_data$PHQ1st, rf_predictions, 
             main = "Actual vs Predicted - Random Forest",
             xlab = "Actual", ylab = "Predicted", col = "blue", pch = 19)
        abline(0, 1, col = "red", lty = 2)
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
        label = train_data$PHQ1st,
        nrounds = input$trees,
        max_depth = 3,
        eta = 0.1,
        objective = "reg:squarederror",
        verbose = 0
      )
      
      gbm_predictions <- predict(gbm_model, test_matrix)
      mse <- mean((test_data$PHQ1st - gbm_predictions)^2)
      rmse <- sqrt(mse)
      
      output$model_output <- renderText({
        paste("Gradient Boosting: MSE =", round(mse, 3), ", RMSE =", round(rmse, 3))
      })
      
      output$varimp_plot <- renderPlot({
        importance <- xgb.importance(feature_names = colnames(train_matrix), model = gbm_model)
        xgb.plot.importance(importance, main = "Variable Importance - Gradient Boosting")
      })
      
      output$roc_plot <- renderPlot({
        plot(test_data$PHQ1st, gbm_predictions, 
             main = "Actual vs Predicted - Gradient Boosting",
             xlab = "Actual", ylab = "Predicted", col = "green", pch = 19)
        abline(0, 1, col = "red", lty = 2)
      })
    }
  })
}

shinyApp(ui, server)
