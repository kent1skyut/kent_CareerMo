$(document).ready(function () {
  // Select the search input field
  const jobSearchInput = $('#jobSearch');

  // Select the suggestions container
  const suggestionsContainer = $('#suggestions');

  // Attach an input event listener to the search input
  jobSearchInput.on('input', function () {
    // Get the user's input
    const query = $(this).val();

    // Make an AJAX request to the server to get job suggestions
    $.ajax({
      method: 'GET',
      url: `/get_job_suggestions?query=${query}`, // Replace with your server endpoint
      success: function (data) {
        // Clear previous suggestions
        suggestionsContainer.empty();

        // Display the new job suggestions
        data.suggestions.forEach(function (suggestion) {
          const suggestionElement = $('<div>').text(suggestion);
          suggestionsContainer.append(suggestionElement);
        });
      },
      error: function (error) {
        console.error('Error:', error);
      },
    });
  });
});
