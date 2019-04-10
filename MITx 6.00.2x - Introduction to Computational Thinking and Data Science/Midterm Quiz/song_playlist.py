#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:02:00 2018

@author: Jake
"""

def song_playlist(songs, max_size):
    """
    songs: list of tuples, ('song_name', song_len, song_size)
    max_size: float, maximum size of total songs that you can fit

    Start with the song first in the 'songs' list, then pick the next 
    song to be the one with the lowest file size not already picked, repeat

    Returns: a list of a subset of songs fitting in 'max_size' in the order 
             in which they were chosen.
    """
    
    #initiate playlist and playlist_size. Copy songs to avail_songs for mutation
    playlist = []
    playlist_size = 0
    avail_songs = songs
    
    #add first song to list if it fits and delete from avail_songs
    if avail_songs[0][2] < max_size:
        playlist.append(avail_songs[0][0])
        playlist_size += avail_songs[0][2]
        del avail_songs[0]
    else:
        return playlist
    
    #sort list by file size
    avail_songs.sort(key=lambda x: x[2])
    
    #walk down avail_songs, add until we reach max size
    for i in range(len(avail_songs)):
        if avail_songs[i][2] < (max_size - playlist_size):
            playlist.append(avail_songs[i][0])
            playlist_size += avail_songs[i][2]
    
    return playlist




songs = [('Roar',4.4, 4.0),('Sail',3.5, 7.7),('Timber', 5.1, 6.9),('Wannabe',2.7, 1.2)]
max_size = 12.2
print(song_playlist(songs, max_size))
songs = [('Roar',4.4, 4.0),('Sail',3.5, 7.7),('Timber', 5.1, 6.9),('Wannabe',2.7, 1.2)]
max_size = 11
print(song_playlist(songs,max_size))