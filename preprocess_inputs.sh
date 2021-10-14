#/bin/bash
#$ -S /bin/bash
set -x -e


ROOT='/home/sadhana-ravikumar/Documents/Sadhana/exvivo_cortex_unet'
INPUTS=$ROOT/inputs
DATADIR=$ROOT/data_csv
CODEDIR=$ROOT/code

SUBJ_TXT=$ROOT/subj_train.txt
IND_ALL="$(cat $SUBJ_TXT)"

function main()
{

  mkdir -p $DATADIR
  process_inputs

  #Preparation
  #PreparationTest

# PreparePulkitData
}

function process_inputs()
{

  FILE=$ROOT/subj_train.txt
  N=$(cat $FILE | wc -l)

  for ((i=1;i<=${N};i++)); do

        LINE=$(cat $FILE | head -n $i | tail -n 1)
        id=$(echo $LINE | cut -d ' ' -f 1)

	echo $id	
	#echo "Normalizing image for  $id"
	IMG=$INPUTS/${id}/${id}_clip_n4.nii.gz
	SEG=$INPUTS/${id}/${id}_multilabel_corrected_seg_whippo.nii.gz
	DMAP=$INPUTS/${id}/${id}_coords-IO.nii.gz

 	IMG_TRIM=$INPUTS/${id}/${id}_trimmed_img.nii.gz
	SEG_TRIM=$INPUTS/${id}/${id}_trimmed_phg.nii.gz
	DMAP_TRIM=$INPUTS/${id}/${id}_trimmed_dmap.nii.gz
	

      #Trim input image to only contain segmented region
	c3d $SEG -trim 0vox -o $SEG_TRIM -thresh -inf inf 1 0 -popas MASK $IMG \
	-push MASK -reslice-identity -as R $IMG -add -push R -times -trim 0vox \
        -shift -1 -o $IMG_TRIM \
	$DMAP -push R -add -push R -times -trim 0vox -shift -1 -o $DMAP_TRIM 
  
	#Downsample the trimmed input image since such a high res is not required. Patch will capture more info
  #	c3d $IMG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_img.nii.gz
  #	c3d $SEG_TRIM -resample 75% -o $INPUTS/${id}/${id}_downsample_phg.nii.gz
	# Post processing to visualize test results

  done
}

function Preparation()
{ 

  N=$(cat $SUBJ_TXT | wc -l)
  rm -rf $DATADIR/split.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $SUBJ_TXT | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $SUBJ_TXT | grep $id)   
  
    IMG=$INPUTS/${id}/${id}_trimmed_img.nii.gz
    SEG=$INPUTS/${id}/${id}_trimmed_phg.nii.gz
    DMAP=$INPUTS/${id}/${id}_trimmed_dmap.nii.gz

    echo "$IMG,$SEG,$idint,"Control",$type,$DMAP" >> $DATADIR/split.csv

  done
}

# Jan 6 - got rid of test dataset. Can cross validate instead to make use of all data
function PreparationTest()
{

  N=$(cat $ROOT/subj_test.txt | wc -l)
  rm -rf $DATADIR/split_test.csv

  for ((i=1;i<=${N};i++)); do

    LINE=$(cat $ROOT/subj_test.txt | head -n $i | tail -n 1)
    id=$(echo $LINE | cut -d ' ' -f 1)
    read dummmy type idint <<< $(cat $ROOT/subj_test.txt | grep $id)

      IMG=$INPUTS/${id}/${id}_trimmed_img.nii.gz
      SEG=$INPUTS/${id}/${id}_trimmed_phg.nii.gz

      echo "$IMG,$SEG,$idint,"Control",$type" >> $DATADIR/split_test.csv

  done
}

function PreparePulkitData()
{

for filename in /home/sadhana-ravikumar/Documents/Sadhana/mtl94_forpulkit/*.nii.gz; do

  file=$(basename $filename)
  id=$(echo $file | cut -d_ -f1)
  echo "$filename,$filename,$id,"Control","test"" >> $DATADIR/test_pulkit.csv

 done

}
main