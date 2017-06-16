/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.android.datafrominternet;

import android.content.Context;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private EditText mSearchBoxEditText;

    private TextView mUrlDisplayTextView;

    private TextView mSearchResultsTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mSearchBoxEditText = (EditText) findViewById(R.id.et_search_box);

        mUrlDisplayTextView = (TextView) findViewById(R.id.tv_url_display);
        mSearchResultsTextView = (TextView) findViewById(R.id.tv_github_search_results_json);
    }

    // Do 2 - 7 in main.xml ///////////////////////////////////////////////////////////////////////
    // COMPLETED (2) Create a main xml called 'main.xml' in the res->main folder
    // COMPLETED (3) Add one main item to your main
    // COMPLETED (4) Give the main item an id of @+id/action_search
    // COMPLETED (5) Set the orderInCategory to 1
    // COMPLETED (6) Show this item if there is room (use app:showAsAction, not android:showAsAction)
    // COMPLETED (7) Set the title to the search string ("Search") from strings.xml
    // Do 2 - 7 in main.xml ///////////////////////////////////////////////////////////////////////


    // COMPLETED(8) Override onCreateOptionsMenu
    // COMPLETED (9) Within onCreateOptionsMenu, use getMenuInflater().inflate to inflate the main
    // COMPLETED (10) Return true to display your main
    @Override
    public boolean onCreateOptionsMenu(Menu menu){
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }
    // COMPLETED (11) Override onOptionsItemSelected
    // COMPLETED (12) Within onOptionsItemSelected, get the ID of the item that was selected
    // COMPLETED (13) If the item's ID is R.id.action_search, show a Toast and return true to tell Android that you've handled this main click
    // COMPLETED (14) Don't forgot to call .show() on your Toast
    // COMPLETED (15) If you do NOT handle the main click, return super.onOptionsItemSelected to let Android handle the main click
    public boolean onOptionsItemSelected(MenuItem item){
        int menuItemThatWasSelected = item.getItemId();
        if(menuItemThatWasSelected == R.id.action_search){
            Context context = MainActivity.this;
            String message = "Search menu was Clicked";
            Toast.makeText(context, message, Toast.LENGTH_LONG).show();
        }
        return super.onOptionsItemSelected(item);
    }
}
