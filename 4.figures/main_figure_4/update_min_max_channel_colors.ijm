// This macro was adapted from Mike Lippincott's original macro to use with the NF1 data: 
// https://github.com/MikeLippincott/Interstellar_Analysis/blob/main/figures/S13/imageJ_macros/channel_change.ijm
macro "Maximum Intensity Projection" {
	//INPUT/OUPUT folders
	inDir=getDirectory("Choose the input folder");
	outDir=getDirectory("Choose the ouput folder");
	myList=getFileList(inDir);  //an array
	start = getTime();
	waitForUser("I solemnly swear I am up to no good");
	// Make an array of png files only
	flist = newArray(0);
	for (i=0; i<myList.length; i++) {
		if (endsWith(myList[i], ".png")) {
			flist = append(flist, myList[i]);
		}
	}

	for (j = 0 ; j < flist.length ; j++ ){
		progress = j / flist.length;
		progress = progress * 100;
		print(progress+"% complete");
		path=inDir+flist[j];
		open(path);
		a = getTitle();


		print(a);
		if (indexOf(a, "d1") >= 0) {
			selectWindow(a);
			run("Blue");
			print("match");

		} else if (indexOf(a, "d2") >= 0) {
			selectWindow(a);
			run("Green");
			print("match");
		} else if (indexOf(a, "d3") >= 0) {
			selectWindow(a);
			run("Magenta");
			print("match");
		} else if (indexOf(a, "d4") >= 0) {
			selectWindow(a);
			run("Red");
			print("match");
		} else {
			print("no matches found");
		}
		
		selectWindow(a);
		a = replace(a, ".png", "");
		saveAs("PNG", outDir+a+".png");
		selectWindow(a+".png");
		close();

	}
	sec = (getTime()-start)/1000;
	min = sec/60;
	hour = min/60;
	print(sec+" seconds");
	print(min+" minutes");
	print(hour+" hours");
	waitForUser("Mischeif Managed");
}


function append(arr, value) {
    arr2 = newArray(arr.length+1);
    for (i=0; i<arr.length; i++)
        arr2[i] = arr[i];
        arr2[arr.length] = value;
    return arr2;
}
