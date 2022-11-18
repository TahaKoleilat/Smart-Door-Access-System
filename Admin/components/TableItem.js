import React from "react";

export default function TableItem(prop) {
  return (
    <>
      <tbody>
        <tr>
          <td data-label="Person">{prop.person}</td>
          <td data-label="Mask">{prop.Mask}</td>
          <td data-label="FaceMatch">{prop.FaceMatch}</td>
          <td data-label="Period">{prop.Period}</td>
        </tr>
      </tbody>
    </>
  );
}
