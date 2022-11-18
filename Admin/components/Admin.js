import React from "react";
import "./Admin.css";
import "./TableItem.js";
import TableItem from "./TableItem.js";
import people from "../visitors.json";
export default function Admin() {
  return (
    <>
      <table>
        <caption>Admin Panel</caption>
        <thead>
          <tr>
            <th scope="col">Person</th>
            <th scope="col">Mask</th>
            <th scope="col">Face Match</th>
            <th scope="col">Period</th>
          </tr>
        </thead>

        {people.people.map((item) => (
          <TableItem
            person={item.person}
            Mask={item.Mask}
            FaceMatch={item.FaceMatch}
            Period={item.Period}
          />
        ))}
      </table>
    </>
  );
}
