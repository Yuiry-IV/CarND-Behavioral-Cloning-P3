import   tensorflow  as       tf 
import   numpy       as       np
import   pandas      as       pd
from     PIL         import   ImageOps
from     PIL         import   Image, ImageDraw
from     PIL         import   ImageEnhance
from     PIL         import   ImageFilter
import   random
import   matplotlib.pyplot as plt
import   time

def plot_loss(history, title):
   """
     plot training curves to file with title
   """
   ax = range( len(history.history['loss']) )
   plt.plot(ax, history.history['loss'], label='loss')
   plt.plot(ax, history.history['val_loss'], label='val_loss')
   #plt.ylim([0, 10])
   plt.xlabel('Epoch')
   plt.ylabel('Error')
   plt.legend()
   plt.grid(True)
   plt.title( title ) 
   plt.savefig( './output/'+ title+'.png' )

def prepocess_image( file_path, needFlip, dropShadow ):
   """
      preprocess singe image
      1. read image from file path
      2. drop random shadow if required 
      3. crop image 
      4. resize 200x66 
      5. convert to array. Output will be uint8 in range [0:255]      
   """
   with tf.keras.preprocessing.image.load_img( file_path ) as image:

      if dropShadow == 1:
         image = image.convert('RGBA')
         shadow_mask = Image.new('RGBA', (320, 160) )
         shadow_x = np.sort( np.random.randint(low=0, high=320, size=4) )
         pdraw = ImageDraw.Draw(shadow_mask)         
         pdraw.polygon( [ (shadow_x[0],0), (shadow_x[1],160), (shadow_x[2], 160), (shadow_x[3], 0) ], fill=(0,0,0,205), outline=None )      
         image.paste(shadow_mask,mask=shadow_mask)
         image = image.convert('RGB')
      
      image = image.crop( (0,69,320,138) )
      image = image.resize( (200, 66) )

      if needFlip == 1:
         image = image.transpose( Image.FLIP_LEFT_RIGHT )
      
      image = tf.keras.preprocessing.image.img_to_array( image, dtype=np.uint8 )        
      return image
# 
def prepare_from_folder( path ) :
   """
      load all images from folder, specified by driving_log.csv
      also configure required data augmentation
      * drop shadow on image 
      * flip image 
      * shift steering angle according to camera position
   """
   data_list=[]

   df = pd.read_csv( path + 'driving_log.csv', sep=',' )

   for i in range( 0, df.shape[0] ) :
      steering = df.iloc[i][3]

      cent   = path + str( df.iloc[i][0] )
      left   = path + str( df.iloc[i][1] ).strip()
      rigth  = path + str( df.iloc[i][2] ).strip()
      
      for shadow in [ 0, 1 ]:
         for flip in [ 0, 1 ]:
            data_list.append( {'steering':steering,      'file_name':cent,  'flip':flip, 'dropshadow':shadow } )
            data_list.append( {'steering':steering+0.21, 'file_name':left,  'flip':flip, 'dropshadow':shadow } )
            data_list.append( {'steering':steering-0.21, 'file_name':rigth, 'flip':flip, 'dropshadow':shadow } )
   
   return data_list


def prepare_all_data( ):
   """
      prepare data from requred folders
   """
    BASE_PATH = '../data/'
    all_data = []
    all_data = all_data + prepare_from_folder( BASE_PATH + 'ref/')
    all_data = all_data + prepare_from_folder( BASE_PATH + 'track_2_1/')
    all_data = all_data + prepare_from_folder( BASE_PATH + 'track_2_2/')
    all_data = all_data + prepare_from_folder( BASE_PATH + 'TR02/')
    all_data = all_data + prepare_from_folder( BASE_PATH + 'TR0129/')
    all_data = all_data + prepare_from_folder( BASE_PATH + 'T203/')
    
    print( 'samples :', len(all_data) )
    
    return all_data

def to_np_array( data, test_run=False ):
   """
      convert data from list with confiration options to numpy array, alos shuffle data
   """
   random.shuffle( data )
   random.shuffle( data )
   random.shuffle( data )

   if test_run :
      print( 'test run. shrink data to 5%' )
      data  = data[0:round(len(data)*.05) ]

   x = np.zeros( ( len(data), 66, 200, 3), dtype=np.uint8 )
   y = np.zeros( ( len(data)            ), dtype=np.float32 )

   for i in range( len(data) ):
      x[i] = prepocess_image( data[i]['file_name'], data[i]['flip'], data[i]['dropshadow']  )
      y[i] = data[i]['steering']
      if data[i]['flip'] == 1:
         y[i] = -1.0 * y[i]
         
      if (i%(len(data)//100)) == 0:
         print( 'loading {:3d}%'.format( i//(len(data)//100) ), end='\r' )

   print( 'loading finsihed.\n' );
   print( 'x :', x.min(), x.max() )
  
   print( 'steering[] orig :', y.min(), y.max() )   
   y = y - y.min()
   print( 'steering[] -min :', y.min(), y.max() )
   y = y / y.max() 
   print( 'steering[] /max :', y.min(), y.max() )
   
   print( x.shape, y.shape )
   return x, y 


def get_model_nv( ) :    
   """
      Create a Nvidia CNN model for automonus driving. Detailed description of model available on https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
   """
   dropout = 0.3
   kernel_regularizer = tf.keras.regularizers.l2( 1e-4 )
   activation = 'relu'
   padding = 'valid'
   
   layers = []
      
   layers.append(  tf.keras.layers.InputLayer( input_shape=( 66, 200, 3) )                                                            )
   layers.append(  tf.keras.layers.experimental.preprocessing.Rescaling(  scale=1./127.5, offset=-1.0  )                              )
   layers.append(  tf.keras.layers.Conv2D( filters=24, kernel_size=(5, 5), strides=( 2, 2), padding='valid', activation=activation )  )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Conv2D( filters=36, kernel_size=(5, 5), strides=( 2, 2), padding='valid', activation=activation )  )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Conv2D( filters=48, kernel_size=(5, 5), strides=( 2, 2), padding='valid', activation=activation )  )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Conv2D( filters=64, kernel_size=(3, 3), strides=( 1, 1), padding='valid', activation=activation )  )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Conv2D( filters=64, kernel_size=(3, 3), strides=( 1, 1), padding='valid', activation=activation )  )
   layers.append(  tf.keras.layers.Flatten( )                                                                                         )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Dense(  100, activation=activation, kernel_regularizer = kernel_regularizer)                       )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Dense(   50, activation=activation, kernel_regularizer = kernel_regularizer )                      )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Dense(   10, activation=activation, kernel_regularizer = kernel_regularizer )                      )
   layers.append(  tf.keras.layers.Dropout( dropout )                                                                                 )
   layers.append(  tf.keras.layers.Dense( 1 )                                                                                         )
   
   model = tf.keras.Sequential( layers, 
         name= 'nvidia_cnn_Dp{:.2f}_kr{:.4f}'.format(dropout, kernel_regularizer.l2)
      )
     
   model.compile(  optimizer = tf.keras.optimizers.Adam( )
               , loss    = tf.keras.losses.MeanSquaredError( )
              )    
   return model


# Load the data
raw_data = prepare_all_data( )
x, y = to_np_array( raw_data, test_run=False )

# use a GPU for trainig
with tf.device('/GPU:0'):
   # prepapre a model check point callback
   callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
     filepath='./saved/model.{val_loss:.4f}.{epoch:02d}.h5',
     monitor='val_loss',
     save_best_only=True
   )

   # create a model and print a model name
   model = get_model_nv()   
   print( model.name )
   
   # run train loop
   history = model.fit(   x = x
            , y = y
            , batch_size=1000
            , epochs = 20
            , validation_split = 0.2
            , shuffle = True
            , callbacks = [ callback_model_checkpoint
                          ]
   )

   # draw history 
   plot_loss( history, model.name + '_{:.4f}'.format(history.history['val_loss'][-1]) )

   # save model summary to file
   with open( './output/' + model.name +  '.txt','w') as fh:
       # Pass the file handle in as a lambda function to make it callable
       model.summary(print_fn=lambda x: fh.write(x + '\n'))

   # plot the model architecture
   tf.keras.utils.plot_model( 
      model, 
      to_file='./output/' + model.name +  '_plt.png', 
      show_shapes=True, 
      show_dtype=False,
      show_layer_names=True, 
      rankdir='TB', 
      expand_nested=True, 
      dpi=120
   )
