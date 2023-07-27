# Import libraries
library(shiny)
library(bslib)
library(httr)
library(broom)
library(irlba)
library(tidyverse)

# Load data
movie_ready = read_csv("movie_1000.csv")
embeddings_similarity = read_csv("embeddings_top1000_similarity_df.csv")

source("setAccountInfo.R")

# Define UI
ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "vapor"),
  tags$head(tags$style(HTML("
                            body {
                              background-color: #333333;
                            }
                            "))), 
  titlePanel(
    HTML("Plot Twist: A New Way to Discover Movies <br> <h4 style='font-size:85%;'> Plot Twist uses AI embeddings to analyze movie plot summaries, generating recommendations based on narrative similarity for a unique film discovery experience. </h4>")
  ),
  
  sidebarLayout(
    sidebarPanel(
      selectizeInput("movie_name", "Enter Movie Name", 
                     choices = sort(movie_ready$title),  # Sort the movie titles
                     selected = NULL, 
                     options = list(maxOptions = length(unique(movie_ready$title)))),
      actionButton("go", "Submit")
    ),
    
    mainPanel(
      h3("Here Are Your Recommendations:", align = "left"),
      tableOutput("recommendations"),
      tags$footer(HTML("Developed by <a href='https://twitter.com/j_franceschini' target='_blank'>@j_franceschini</a>"))  # Add a footer with a link
    )
  )
)

# Define server logic
server <- function(input, output) {
  observeEvent(input$go, {
    # Movie name entered by user
    movie_name <- input$movie_name
    
    # Find rowid of this movie in the similarity matrix
    movie_index <- which(movie_ready$title == movie_name)
    
    if(length(movie_index) == 0) {
      output$recommendations <- renderTable(NULL)
      return()
    }
    
    # Fetch similarity scores for this movie
    similarity_scores <- embeddings_similarity[movie_index, ]
    
    # Convert to a named vector: names are movie titles, values are similarity scores
    similarity_vector <- setNames(unlist(similarity_scores), movie_ready$title)
    
    # Sort by similarity score
    sorted_vector <- sort(similarity_vector, decreasing = TRUE)
    
    # Select top N movies to recommend (exclude first, as it's the same movie)
    top_n <- 10
    recommended_movies <- head(sorted_vector[-1], top_n)  # Skip the first one
    
    # Scale the similarity scores to 0-100
    recommended_movies <- round(recommended_movies * 100)  # Round to the nearest integer
    
    # Get box office data for the recommended movies
    box_office <- movie_ready$box_office[match(names(recommended_movies), movie_ready$title)]
    box_office_m <- paste0("$", round(box_office / 1000000, 0), "M")
    
    # Create a data frame of recommendations
    recommendations <- data.frame(
      Movie = names(recommended_movies),
      Similarity = paste0(recommended_movies, "%"),  # Convert to percentage format
      'Worldwide Box Office' = box_office_m,
      check.names = FALSE
    )
    
    # Render recommendations as a table in main panel
    output$recommendations <- renderTable({
      recommendations
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
