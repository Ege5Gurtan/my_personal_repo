  
  const optional_columns = ["OffsetX","OffsetY","SizeX","SizeY","rotation","scaling","centered","mirrorx"]



function get_column_values(hot,column_index){
let columnData = [];
let numberOfRows = hot.countRows();
for (let i = 0; i < numberOfRows; i++) {
    let cellData = hot.getDataAtCell(i, column_index); // 0 is the column index for the first column
    columnData.push(cellData);
}
  return columnData
}
function check_mark_column(cell_value){
    if (cell_value===""){
      return false
    }
}
  function check_reference(cell_value,mentioned_marks,td,row_num){
    if (cell_value!=='')
    {
      if (cell_value!==null)
      {
      if (!cell_value.includes('/'))
      {
          if (!mentioned_marks.includes(cell_value))
          {
          td.style.backgroundColor = 'red';
          console.log('The cell name has not been defined!')
          }
          else{
            const mark_index = mentioned_marks.indexOf(cell_value)
            if (mark_index > row_num){
              console.log('Cell name has been referenced before the assignment!')
              td.style.backgroundColor = 'red';
            }
          }
      }
    }
    }
  }

function check_cdx_cdy(cell_value,td,reference_value){

  if (reference_value!==''){
    if (reference_value!==null){
  
  if (reference_value.includes('/')){
      const number = parseFloat(cell_value);
      if (!Number.isFinite(number)){
        td.style.backgroundColor = 'red';
        console.log('Please provide a finite number for layer references.')
      }
  }

  else{
    if (cell_value!==null){
      td.style.backgroundColor = 'red';
      console.log('CD values should be left empty for cell references.')
    }
    if (cell_value ==''){
      td.style.backgroundColor = 'white';
    }
  }
}
  }
}


function myCustomRenderer(instance, td, row, col, prop, value, cellProperties) {
    Handsontable.renderers.TextRenderer.apply(this, arguments);
        
        // check reference column
        if (col == column_header_indices['reference']){ 
          let declared_marks = get_column_values(instance,column_header_indices['Mark'])
          check_reference(value,declared_marks,td,row)
        }

        //check CdX and CdY columns
        if (col == column_header_indices['CdX'] || col == column_header_indices['CdY'])
        {
          let reference_values = get_column_values(instance,column_header_indices['reference'])
          let current_reference_value = reference_values[row]
          check_cdx_cdy(value,td,current_reference_value)
        }
}
  function initializeHandsontable(row_num,col_num,column_names,data) {
    const container = document.querySelector('#csv_table');

    let renderers = column_names.map(element =>{
      return {renderer: myCustomRenderer}
    })

    handson_table = new Handsontable(container, {
      startRows: row_num,
      startCols: col_num,
      minRows: row_num,
      minCols: col_num,
      data: data,
      rowHeaders: true,
      colHeaders: true,
      filters: true,
      dropdownMenu: false,
      manualRowResize: true,
      manualColumnResize: true,
      colHeaders : column_names,
      columns: renderers, 

    });
    return handson_table
  }
  function add_checkbox_event_listener(header_name,handson_table){
    const check_box = document.querySelector('#'+header_name);
    check_box.addEventListener('change', () => {
        var current_column_headers = handson_table.getColHeader();
        if (check_box.checked){
            updated_column_headers = [...current_column_headers];
            if (!added_column_index.hasOwnProperty(header_name)){
            added_column_index[header_name] = Object.keys(added_column_index).length
            updated_column_headers.splice(column_names.length+added_column_index[header_name], 0, header_name);
        }
        else{
            updated_column_headers.splice(column_names.length+added_column_index[header_name], 1, header_name);
        }
        }
        else{
            updated_column_headers = [...current_column_headers];
            const indexToReplace = updated_column_headers.indexOf(header_name);
            updated_column_headers[column_names.length+added_column_index[header_name]] = "removed";
        }

        
        let updated_renderers = updated_column_headers.map(element =>{
          return {renderer: myCustomRenderer}
        })
        
        handson_table.updateSettings({
            colHeaders: updated_column_headers,
            minCols: updated_column_headers.length,
            columns: updated_renderers
        });
    });
    
}
function exportToCSV(updated_column_headers, handson_table) {
    
    const selectedColumns = updated_column_headers.map((name, index) => name !== 'removed' ? index : -1).filter(index => index !== -1);
    
    
    // Filter the updated column headers based on the selected columns
    const filteredHeaders = updated_column_headers.filter((header, index) => selectedColumns.includes(index));

    // Get the data from the Handsontable instance
    const data = handson_table.getData();
  
    // Filter the data to include only the selected columns
    const filteredData = data.map(row => selectedColumns.map(index => row[index]));
  
    console.log(data)
    console.log(filteredData)
  

    // Create a temporary Handsontable instance to use the export plugin
    var tempContainer = document.createElement('div');
    document.body.appendChild(tempContainer);
  
    var mapped_columns = selectedColumns.map(index => ({ data: index }))
    var tempHot = new Handsontable(tempContainer, {
      data: data,
      colHeaders: filteredHeaders,
      columns:mapped_columns ,
    
    });
  
    // Use the export plugin to download the file
    var exportPlugin = tempHot.getPlugin('exportFile');
    exportPlugin.downloadFile('csv', {
      filename: 'Handsontable_Selected_Columns_CSV',
      columnHeaders: true,
    });
  
    // Clean up the temporary Handsontable instance
    document.body.removeChild(tempContainer);
  }




