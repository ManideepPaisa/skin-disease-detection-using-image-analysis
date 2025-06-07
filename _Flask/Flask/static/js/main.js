$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);

                // Extract information from the response
                const disease = data.disease;
                const confidence = data.confidence;
                const suggestions = data.suggestions;
                const medications = data.medications;
                const note = data.note;  // Extract the note

                // Create the result HTML
                const resultHTML = `
                    <h4>Prediction: ${disease}</h4>
                    <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(2)}%</p>
                    <h5>Suggestions:</h5>
                    <ul>${suggestions.map(s => `<li>${s}</li>`).join('')}</ul>
                    <h5>Medications:</h5>
                    <ul>${medications.map(m => `<li><strong>${m.name}:</strong> ${m.details}</li>`).join('')}</ul>
                    <p><strong>Note:</strong> ${note}</p>  <!-- Add the note here -->
                `;

                // Display the result HTML
                $('#result').html(resultHTML);
                console.log('Success!');
            },
            error: function (xhr, status, error) {
                // Handle errors here
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html('<p>Error in prediction. Please try again.</p>');
                console.log('Error:', error);
            }
        });
    });
});
