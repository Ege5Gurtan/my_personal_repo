const column_names = ["Mark","reference","CdX","CdY","PitchX","PitchY","Nx","Ny"]
const added_column_index = {}

let column_header_indices = {};
for (let i = 0; i < column_names.length; i++) {
    column_header_indices[column_names[i]] = i;
}



const all_column_names = [...column_names ,...optional_columns,"removed"]
let updated_column_headers;

const data = [
    [],
  ];

handson_table = initializeHandsontable(10,column_names.length,column_names,data)

//document.getElementById('export-button').addEventListener('click', exportToCSV);
//document.getElementById('export-button').addEventListener('click', exportSpecifiedColumnsToCSV);



document.getElementById('Options').onclick = function() {
    const sidePanel = document.querySelector('#sidePanel');
    
    if (sidePanel.style.right === '0px') {
        sidePanel.style.right = '-250px';
    } else {
        sidePanel.style.right = '0px';
    }
};



const all_options = document.querySelector("#all_options");
optional_columns.forEach(option_element => {
    const current_element = `<label><input type="checkbox" value="${option_element}" id=${option_element}>${option_element}</label><br>`;
    all_options.insertAdjacentHTML("beforeend", current_element);
});


optional_columns.forEach(current_column =>{
    add_checkbox_event_listener(current_column,handson_table)
})


updated_column_headers = [...column_names]
document.getElementById('export-button').addEventListener('click', (event) => exportToCSV(updated_column_headers, handson_table));


