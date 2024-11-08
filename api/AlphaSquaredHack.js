// This is more of an Hack than anybody else, log into AlphaSquare, open the diagram and fire up this script. 
// It will copy the data into a new tab.

// Access the data arrays
var xData = Highcharts.charts[0].series[1].processedXData;
var yData = Highcharts.charts[0].series[1].processedYData;

// Ensure the arrays have the same length
if (xData.length !== yData.length) {
    console.error('Data arrays have mismatched lengths.');
} else {
    // Initialize the CSV content with headers
    var csvContent = "Date;Risk\n";

    // Loop through the data and build the CSV string
    for (var i = 0; i < xData.length; i++) {
        var timestamp = xData[i];
        var riskValue = yData[i];

        // Convert timestamp to a readable date
        var date = new Date(timestamp);
        var formattedDate = date.toISOString().split('T')[0];

        // Append to CSV content
        csvContent += formattedDate + ';' + riskValue + '\n';
    }

    // Create a data URL
    var dataUrl = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent);

    // Open the data URL in a new tab
    window.open(dataUrl);

    // Inform the user
    console.log('Data has been opened in a new tab.');
}
